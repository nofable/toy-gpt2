import torch

def main() -> None:
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because current pytorch was not built with mps enabled.")
        else:
            print("MPS not available because this mac doesn't support it")

    else:
        mps_device = torch.device("mps")
        x = torch.rand(4, 4, device=mps_device)
        y = torch.rand(4, 4, device=mps_device)
        result = x @ y
        print(result.device)
