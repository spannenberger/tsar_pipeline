import torch
import sys

if __name__ == "__main__":
    num_fold = sys.argv[1]
    checkpoint = torch.load(f"crossval_log/{num_fold}/checkpoints/best.pth")
    print("Fold metric:", checkpoint["valid_metrics"][checkpoint["main_metric"]])