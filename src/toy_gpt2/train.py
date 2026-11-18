"""Train a GPT. Defaults = the GPT-2 124M run on FineWeb-Edu (modern stack: pre-LN, RMSNorm, RoPE, SwiGLU).

  uv run python -u -m toy_gpt2.train --name gpt2 --max_hours 5.3      # one GPU
  uv run torchrun --standalone --nproc_per_node 2 -m toy_gpt2.train --name gpt2 --max_hours 9.5   # DDP
  uv run python -u -m toy_gpt2.train --name gpt2 --resume             # continue after a crash
  uv run python -u -m toy_gpt2.train --name smoke --data tinystories --n_layer 2 --n_head 2 --n_embd 128 \
      --block_size 128 --batch_size 8 --total_batch_tokens 1024 --max_iters 50 --warmup_iters 10

Writes runs/<name>/: ckpt.pt (model + optimizer + data position), log.csv (evals), steps.csv, samples.txt
"""

import argparse
import csv
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import tiktoken
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from toy_gpt2.data import ShardLoader, shard_paths
from toy_gpt2.gpt import GPT, GPTConfig, get_device
from toy_gpt2.schedule import get_lr

PROMPTS = {"finewebedu": "The most important thing to know about", "tinystories": "Once upon a time"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="gpt2")
    p.add_argument("--data", choices=["finewebedu", "tinystories"], default="finewebedu")
    p.add_argument("--resume", action="store_true")

    p.add_argument("--vocab_size", type=int, default=50304)
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=12)
    p.add_argument("--n_embd", type=int, default=768)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="rmsnorm")
    p.add_argument("--norm_position", choices=["pre", "post"], default="pre")
    p.add_argument("--pos_emb", choices=["learned", "rope"], default="rope")
    p.add_argument("--mlp", choices=["gelu", "swiglu"], default="swiglu")

    p.add_argument("--batch_size", type=int, default=64, help="sequences per micro-batch")
    p.add_argument("--total_batch_tokens", type=int, default=524_288, help="tokens per optimizer step")
    p.add_argument("--max_tokens", type=float, default=10e9)
    p.add_argument("--max_iters", type=int, default=None, help="overrides --max_tokens")
    p.add_argument("--max_hours", type=float, default=None, help="shrink the run to fit this wall-clock budget")
    p.add_argument("--warmup_iters", type=int, default=700)
    p.add_argument("--max_lr", type=float, default=6e-4)
    p.add_argument("--min_lr", type=float, default=None, help="default: max_lr / 10")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None, help="default: on for CUDA")
    p.add_argument("--peak_tflops", type=float, default=None, help="bf16 peak for MFU; default from GPU name")
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--ckpt_interval", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--hellaswag", action=argparse.BooleanOptionalAction, default=None, help="default: on for CUDA")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--stop_after", type=int, default=None, help="testing: save and exit after this step")
    args = p.parse_args(argv)
    if args.min_lr is None:
        args.min_lr = args.max_lr / 10
    return args


def configure_optimizer(model: GPT, args, device: str) -> torch.optim.AdamW:
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": args.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=args.max_lr, betas=(0.9, 0.95), eps=1e-8, fused=device.startswith("cuda"))


def flops_per_token(model: GPT) -> int:
    cfg = model.cfg
    n = sum(p.numel() for p in model.parameters())
    if model.wpe is not None:
        n -= model.wpe.weight.numel()
    return 6 * n + 12 * cfg.n_layer * cfg.n_embd * cfg.block_size


class Dist:
    def __init__(self):
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.on = self.world > 1
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.master = self.rank == 0
        if self.on:
            cuda = torch.cuda.is_available()
            self.device = f"cuda:{self.local_rank}" if cuda else "cpu"
            if cuda:
                torch.cuda.set_device(self.device)
            dist.init_process_group("nccl" if cuda else "gloo")
        else:
            self.device = get_device()

    def mean(self, t: torch.Tensor) -> torch.Tensor:
        if self.on:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t = t / self.world
        return t

    def from_master(self, value: int) -> int:
        if not self.on:
            return value
        t = torch.tensor([value], dtype=torch.long, device=self.device)
        dist.broadcast(t, src=0)
        return int(t.item())

    def any(self, flag: bool) -> bool:
        return self.mean(torch.tensor(float(flag), device=self.device)).item() > 0

    def close(self) -> None:
        if self.on:
            dist.barrier()
            dist.destroy_process_group()


@torch.no_grad()
def val_loss(model, loader: ShardLoader, n_batches: int, autocast, d: Dist) -> float:
    model.eval()
    loader.reset()
    total = torch.zeros((), device=d.device)
    per_rank = max(1, n_batches // d.world)
    for _ in range(per_rank):
        x, y = loader.next_batch()
        with autocast:
            total += model(x, y)[1].float()
    model.train()
    return d.mean(total / per_rank).item()


def fit_max_iters(step: int, max_iters: int, warmup_iters: int, sec_per_step: float, remaining_s: float,
                  overhead: float = 1.02) -> int:
    fit = step + int(remaining_s / (sec_per_step * overhead))
    return min(max_iters, max(fit, warmup_iters + 1))


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    d = Dist()
    say = (lambda *a: print(*a, flush=True)) if d.master else (lambda *a: None)
    out_dir = Path("runs") / args.name
    ckpt_path = out_dir / "ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False) if args.resume else None
    if ckpt is not None:
        resume_flag, args = True, argparse.Namespace(**ckpt["args"])
        args.resume, args.stop_after = resume_flag, None
    if d.master:
        out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = d.device
    cuda = device.startswith("cuda")
    if args.max_hours is not None and not cuda:
        sys.exit(f"--max_hours is for the paid GPU run, but the device is {device}: is CUDA working?")
    use_compile = cuda if args.compile is None else args.compile
    run_hellaswag = cuda if args.hellaswag is None else args.hellaswag
    if cuda:
        torch.set_float32_matmul_precision("high")
        gpu = torch.cuda.get_device_name()
        if args.peak_tflops is None:
            args.peak_tflops = 756.0 if "PCIe" in gpu else 312.0 if "A100" in gpu else 989.0
        say(f"gpu={gpu} x{d.world} torch={torch.__version__} cuda={torch.version.cuda} "
            f"peak={args.peak_tflops} TFLOPS per GPU")
    elif args.peak_tflops is None:
        args.peak_tflops = 989.0
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else nullcontext()

    tokens_per_micro = args.batch_size * args.block_size * d.world
    assert args.total_batch_tokens % tokens_per_micro == 0, "total_batch_tokens must be a multiple of B*T*GPUs"
    grad_accum = args.total_batch_tokens // tokens_per_micro

    train_loader = ShardLoader(shard_paths(args.data, "train"), args.batch_size, args.block_size, device,
                               d.rank, d.world)
    val_loader = ShardLoader(shard_paths(args.data, "val"), args.batch_size, args.block_size, device,
                             d.rank, d.world)

    cfg = GPTConfig(
        vocab_size=args.vocab_size, block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head,
        n_embd=args.n_embd, dropout=args.dropout, norm=args.norm, norm_position=args.norm_position,
        pos_emb=args.pos_emb, mlp=args.mlp,
    )
    raw_model = GPT(cfg).to(device)
    opt = configure_optimizer(raw_model, args, device)

    max_iters = args.max_iters or math.ceil(args.max_tokens / args.total_batch_tokens)
    start_step, elapsed_before, calibrated = 0, 0.0, args.max_hours is None
    if ckpt is not None:
        raw_model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        train_loader.load_state(ckpt["loader"])
        start_step, max_iters = ckpt["step"], ckpt["max_iters"]
        elapsed_before, calibrated = ckpt["elapsed_s"], ckpt["calibrated"]
        say(f"resumed from step {start_step} ({elapsed_before / 3600:.2f} h already used)")
    del ckpt

    model = torch.compile(raw_model) if use_compile else raw_model
    if d.on:
        model = DDP(model, device_ids=[d.local_rank] if cuda else None)
    fpt = flops_per_token(raw_model)
    n_params = sum(p.numel() for p in raw_model.parameters())
    say(f"device={device} gpus={d.world} compile={use_compile} params={n_params / 1e6:.2f}M "
        f"grad_accum={grad_accum} tokens/step={args.total_batch_tokens:,} max_iters={max_iters} "
        f"(~{max_iters * args.total_batch_tokens / 1e9:.2f}B tokens) "
        f"train tokens available={train_loader.total_tokens() / 1e9:.2f}B")

    enc = tiktoken.get_encoding("gpt2")
    prompt = torch.tensor([enc.encode(PROMPTS[args.data])], device=device)
    if d.master:
        mode = "a" if args.resume else "w"
        log_f, steps_f = open(out_dir / "log.csv", mode, newline=""), open(out_dir / "steps.csv", mode, newline="")
        log, steps_log = csv.writer(log_f), csv.writer(steps_f)
        if not args.resume:
            log.writerow(["step", "lr", "train_loss", "val_loss", "grad_norm", "tok_per_s", "elapsed_s"])
            steps_log.writerow(["step", "loss", "lr", "grad_norm", "tok_per_s", "mfu", "elapsed_s"])

    t_start = time.time()

    def elapsed() -> float:
        return elapsed_before + time.time() - t_start

    def checkpoint(next_step: int, val: float | None) -> None:
        if not d.master:
            return
        save_checkpoint(ckpt_path, {
            "model": raw_model.state_dict(), "optimizer": opt.state_dict(), "config": asdict(cfg),
            "args": vars(args), "step": next_step, "max_iters": max_iters, "loader": train_loader.state(),
            "elapsed_s": elapsed(), "calibrated": calibrated, "val_loss": val,
        })
        say(f"saved {ckpt_path} at step {next_step}")

    def close_logs() -> None:
        if d.master:
            log_f.close()
            steps_f.close()

    calib_from, calib_t0 = start_step + 20, None
    recent_losses: list[float] = []
    last_val, t_last, tok_s, grad_norm, lr = None, time.time(), 0.0, 0.0, 0.0

    step = start_step
    while step <= max_iters:
        last_step = step == max_iters
        if step % args.eval_interval == 0 or last_step:
            last_val = val_loss(model, val_loader, args.eval_batches, autocast, d)
            if d.master:
                train_avg = sum(recent_losses) / len(recent_losses) if recent_losses else float("nan")
                recent_losses = []
                say(f"--- step {step}: val {last_val:.4f}  train {train_avg:.4f}  ({elapsed() / 3600:.2f} h)")
                log.writerow([step, lr, train_avg, last_val, float(grad_norm), round(tok_s), round(elapsed())])
                log_f.flush()
                raw_model.eval()
                with autocast:
                    out = raw_model.generate(prompt, 64, temperature=0.8, top_k=50, vocab_limit=enc.n_vocab)
                raw_model.train()
                sample = enc.decode(out[0].tolist())
                say(sample.replace("\n", " "))
                with open(out_dir / "samples.txt", "a") as f:
                    f.write(f"--- step {step} (val {last_val:.4f}) ---\n{sample}\n\n")
        if last_step:
            break

        lr = get_lr(step, args.max_lr, args.min_lr, args.warmup_iters, max_iters)
        for group in opt.param_groups:
            group["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for micro in range(grad_accum):
            x, y = train_loader.next_batch()
            sync = nullcontext() if not d.on or micro == grad_accum - 1 else model.no_sync()
            with sync:
                with autocast:
                    _, loss = model(x, y)
                loss = loss / grad_accum
                loss_sum += loss.detach()
                loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_interval == 0:
            loss_val = d.mean(loss_sum.clone()).item()
            if not math.isfinite(loss_val):
                say(f"loss is {loss_val} at step {step}; stopping without overwriting the checkpoint")
                sys.exit(1)
            recent_losses.append(loss_val)
            now = time.time()
            steps_done = args.log_interval if step > start_step else 1
            tok_s = steps_done * args.total_batch_tokens / (now - t_last)
            t_last = now
            mfu = fpt * tok_s / (args.peak_tflops * 1e12 * d.world)
            mem = f" | mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if cuda else ""
            say(f"step {step:6d} | loss {loss_val:.4f} | lr {lr:.2e} | norm {float(grad_norm):.2f} | "
                f"{tok_s / 1e3:.0f}k tok/s | mfu {mfu:.1%}{mem}")
            if d.master:
                steps_log.writerow([step, round(loss_val, 4), lr, round(float(grad_norm), 3), round(tok_s),
                                    round(mfu, 4), round(elapsed())])
                steps_f.flush()
            if args.max_hours is not None and step + 1 < max_iters:
                if d.from_master(int(elapsed() > args.max_hours * 3600)):
                    say(f"over the time budget at step {step + 1}; ending the run here")
                    max_iters = step + 1

        if not calibrated:
            if step == calib_from:
                calib_t0 = time.time()
            elif step == calib_from + 50:
                sec_per_step = (time.time() - calib_t0) / 50
                remaining = args.max_hours * 3600 - elapsed()
                max_iters = d.from_master(
                    fit_max_iters(step + 1, max_iters, args.warmup_iters, sec_per_step, remaining))
                calibrated = True
                say(f"time budget {args.max_hours} h at {sec_per_step:.2f} s/step -> max_iters={max_iters} "
                    f"(~{max_iters * args.total_batch_tokens / 1e9:.2f}B tokens)")

        step += 1
        if step % args.ckpt_interval == 0:
            if d.any(not (math.isfinite(loss_sum.item()) and math.isfinite(float(grad_norm)))):
                say(f"non-finite loss/grad at step {step - 1}; keeping the previous checkpoint")
                sys.exit(1)
            checkpoint(step, last_val)
        if args.stop_after is not None and step == args.stop_after:
            checkpoint(step, last_val)
            close_logs()
            d.close()
            return

    checkpoint(max_iters, last_val)
    close_logs()
    say(f"done in {elapsed() / 3600:.2f} h. final val {last_val:.4f}. outputs in {out_dir}/")
    d.close()

    if run_hellaswag and d.master:
        from toy_gpt2.hellaswag import evaluate_model

        acc = evaluate_model(raw_model, device, out_dir / "hellaswag.json")
        say(f"HellaSwag acc_norm {acc:.4f} (GPT-2 124M: 0.2955)")


if __name__ == "__main__":
    main()
