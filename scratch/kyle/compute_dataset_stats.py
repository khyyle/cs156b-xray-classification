from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from radiology_cls.data import load_image, load_train_df

N_CHANNELS = 1


class RawImageDataset(Dataset):
    """Load images as single-channel tensors with no normalization."""

    def __init__(self, paths: list[str]):
        self.paths = [p for p in paths if Path(p).exists()]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = load_image(self.paths[idx])
        to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        return to_tensor(img)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-samples", type=int, default=0,
                   help="Cap the number of images (0 = all).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    df = load_train_df()
    if args.max_samples > 0:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=args.seed)

    ds = RawImageDataset(df["image_path"].tolist())
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    print(f"Computing stats over {len(ds)} images...")

    mean = torch.zeros(N_CHANNELS)
    std = torch.zeros(N_CHANNELS)

    for img in tqdm(loader, unit="img"):
        for i in range(N_CHANNELS):
            # dataloader adds batch dimension
            mean[i] += img[:, i, :, :].mean()
            std[i] += img[:, i, :, :].std()

    mean.div_(len(ds))
    std.div_(len(ds))

    print(f"\nGRAYSCALE_MEAN = ({mean[0]:.6f},)")
    print(f"GRAYSCALE_STD  = ({std[0]:.6f},)")


if __name__ == "__main__":
    main()
