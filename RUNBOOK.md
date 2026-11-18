# Runbook: the GPT-2 124M cloud run

One Vast.ai H100, a dry run then the real run on the **same box**. Budget: $20 prepaid. Costs and sizing: [SIZING.md](SIZING.md).

## 0. Before renting (on the Mac)

- [ ] All tests pass: `uv run pytest -q`
- [ ] Everything committed: `git status` is clean
- [ ] Vast.ai account with **$20 credit loaded** (the cap is enforced by Vast)
- [ ] SSH public key added in Vast → Account → SSH Keys (`cat ~/.ssh/id_ed25519.pub`; create one with `ssh-keygen -t ed25519` if missing)

## 1. Rent the box

Search filters:

| Filter | Value | Why |
|---|---|---|
| GPU | 1× **H100 SXM** | PCIe is ~25% slower |
| Max CUDA | **≥ 13.0** | our PyTorch needs NVIDIA driver ≥ 580 |
| Disk | **60 GB** (set at rent time, can't grow later) | 20 GB data + ~8 GB PyTorch + 3 GB checkpoints |
| Type | **On-demand** (not interruptible) | an interruptible box can be taken away mid-run |
| Reliability | > 99% | |
| Download speed | ≥ 1 Gbps | 20 GB of data to download |
| Image | Vast's PyTorch or CUDA Ubuntu image | the script installs its own Python and PyTorch |

- [ ] Note the **price per hour**. If it's above ~$2.60/h, reduce the real run's hours in step 5 so the total stays ≤ $20.
- [ ] Rent it. Copy the SSH command from the instance card (host and port).

## 2. Copy the repo up (from the Mac)

```bash
cd ~/gh/toy-gpt2
rsync -avz -e "ssh -p <PORT>" --exclude .venv --exclude data --exclude runs ./ root@<HOST>:/root/toy-gpt2/
ssh -p <PORT> root@<HOST>
```

## 3. Dry run (~15 min, ~$1)

```bash
cd /root/toy-gpt2
bash scripts/cloud_run.sh 0.25 1 dry --ckpt_interval 100
tail -f runs/dry/stdout.log
```

Check in the log (Ctrl-C stops `tail`, not the training):

- [ ] `nvidia-smi` line: H100, driver ≥ 580
- [ ] `gpu=NVIDIA H100 ... cuda=13.x` and `device=cuda compile=True`
- [ ] No compile error in the first minute
- [ ] Step 0 grad norm may be 15–25; it should fall to ~2–3 within a few steps
- [ ] Around step 70: `time budget ... -> max_iters=...`
- [ ] **tok/s** and **mfu**: write them down (estimate: ~460k tok/s, ~40%)
- [ ] **mem** well under 80 GB (if it crashes with out-of-memory: add `--batch_size 32`)
- [ ] `saved runs/dry/ckpt.pt at step 100`

## 4. Resume test

After the step-100 checkpoint:

```bash
pkill -f toy_gpt2.train
bash scripts/cloud_run.sh 0.25 1 dry
tail -f runs/dry/stdout.log
```

- [ ] Log says `resumed from step 100 (...)` and the loss carries on from where it was (no jump back to ~11)
- [ ] Let it finish: `done in ...` then `HellaSwag acc_norm ...` (expect ~0.25–0.27 after only 15 minutes)

**Stop here and paste the dry-run log to Claude before continuing.**

## 5. Real run (~5.3 h, ~$13.50)

```bash
uv run python scripts/download_finewebedu.py --shards 99
bash scripts/cloud_run.sh 5.3 99 gpt2
tail -f runs/gpt2/stdout.log
```

- [ ] Download finishes (~5–10 min) with no errors
- [ ] Step 70: note the fitted `max_iters` and token count
- [ ] Safe to disconnect: training runs in the background (`nohup`)

Checking in later:

```bash
ssh -p <PORT> root@<HOST>
cd /root/toy-gpt2 && tail -n 30 runs/gpt2/stdout.log
nvidia-smi
```

- [ ] Loss falling, val logged every 500 steps, a checkpoint every 1000 steps
- [ ] Optional insurance: copy a mid-run checkpoint to the Mac (see step 6)

**If it crashes:** read the end of `runs/gpt2/stdout.log`, then rerun `bash scripts/cloud_run.sh 5.3 99 gpt2`. It resumes from the last checkpoint by itself.

## 6. Bring the results home (from the Mac)

When the log shows `HellaSwag acc_norm ...`:

```bash
cd ~/gh/toy-gpt2
rsync -avz -e "ssh -p <PORT>" root@<HOST>:/root/toy-gpt2/runs/gpt2 runs/
rsync -avz -e "ssh -p <PORT>" root@<HOST>:/root/toy-gpt2/runs/dry runs/
ls -la runs/gpt2/
```

- [ ] `ckpt.pt` (~1.5 GB), `log.csv`, `steps.csv`, `samples.txt`, `stdout.log`, `hellaswag.json` are all on the Mac
- [ ] It works locally: `uv run python -m toy_gpt2.sample --run gpt2 --prompt "The history of"`

## 7. Shut down

- [ ] **Destroy** the instance in Vast (not just stop it: a stopped instance still charges for disk)
- [ ] Check the remaining Vast credit and note the total spent
