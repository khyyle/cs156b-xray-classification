"""
Bundle the training CSV and images into a zip for teammates to do EDA locally.

Run on the machine that holds DATA_ROOT:

    python scripts/bundle_data.py
    python scripts/bundle_data.py --max-samples 1000

The zip preserves the Path column structure, so after extracting:

    extracted_dir/
        train2023.csv
        train/pid50512/study1/view1_frontal.jpg
        ...

    df = pd.read_csv("extracted_dir/train2023.csv")
    img = Image.open(f"extracted_dir/{df['Path'].iloc[0]}")
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from radiology_cls.data import load_train_df

DEFAULT_SIZE = 256
DEFAULT_SAMPLES = 5000


def main() -> None:
    p = argparse.ArgumentParser(description="Bundle train data into a zip.")
    p.add_argument("--output", type=Path, default=Path("train_bundle.zip"))
    p.add_argument("--max-samples", type=int, default=DEFAULT_SAMPLES)
    p.add_argument("--image-size", type=int, default=DEFAULT_SIZE,
                   help="Resize images to this square size (default 256)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = load_train_df()
    df = df.sample(n=min(args.max_samples, len(df)), random_state=args.seed)
    print(f"Sampled {len(df)} rows (seed={args.seed})")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    image_paths = df["image_path"].tolist()
    arc_names = df["Path"].tolist()

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        csv_tmp = args.output.with_suffix(".csv")
        df.drop(columns=["image_path"]).to_csv(csv_tmp, index=False)
        zf.write(csv_tmp, "train2023.csv")
        csv_tmp.unlink()

        skipped = 0
        for src, arcname in tqdm(zip(image_paths, arc_names), total=len(df), unit="img"):
            if not Path(src).exists():
                skipped += 1
                continue
            img = Image.open(src).convert("L")
            img = img.resize((args.image_size, args.image_size), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            zf.writestr(arcname, buf.getvalue())

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Done: {args.output} ({size_mb:.1f} MB, {len(df) - skipped} images, {skipped} skipped)")


if __name__ == "__main__":
    main()
