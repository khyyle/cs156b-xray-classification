"""
Pre-resize all training and test images to a target resolution and save
them as small JPEGs in a local cache directory. Loading a 256x256 JPEG
(~8 KB) from cache is much faster than decoding a full-resolution
2022x2022 JPEG on every epoch. The cache has the same structure as the original dataset
and ChestXrayDataset checks for a cached version before loading the original, i.e.,
    cache/256/train/pid50512/study1/view1_frontal.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image, UnidentifiedImageError  # type: ignore
from tqdm import tqdm

from radiology_cls.data import DEFAULT_CACHE_ROOT, load_train_df, load_test_df
from radiology_cls.settings import DATA_ROOT


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def _resize_one(args: tuple[str, str, int, int]) -> tuple[str, str]:
    """Resize a single image and return (status, path)."""
    src, dst, size, quality = args
    dst_path = Path(dst)
    was_corrupt = False
    if dst_path.exists():
        if _is_valid_image(dst_path):
            return "exists", dst
        was_corrupt = True
        dst_path.unlink()

    tmp_path = dst_path.with_suffix(f"{dst_path.suffix}.tmp")
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.unlink(missing_ok=True)
        with Image.open(src) as img:
            resized = img.convert("L").resize((size, size), Image.LANCZOS)
            resized.save(tmp_path, format="JPEG", quality=quality)
        tmp_path.replace(dst_path)
        return ("repaired" if was_corrupt else "created"), dst
    except Exception as e:
        print(f"SKIP {src}: {e}")
        tmp_path.unlink(missing_ok=True)
        return "error", src


def main() -> None:
    p = argparse.ArgumentParser(description="Cache pre-resized images for faster training.")
    p.add_argument("--size", type=int, default=256, help="Target image size (square).")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100).")
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--skip-test", action="store_true", help="Only cache training images.")
    args = p.parse_args()

    cache_dir = args.cache_root / str(args.size)
    print(f"Cache dir: {cache_dir}")
    print(f"Size: {args.size}x{args.size}, quality: {args.quality}")

    train_df = load_train_df()
    rel_paths = train_df["Path"].tolist()

    if not args.skip_test:
        test_df = load_test_df()
        rel_paths += test_df["Path"].tolist()

    work = []
    for rel in rel_paths:
        src = str(DATA_ROOT / rel)
        dst = str(cache_dir / rel)
        work.append((src, dst, args.size, args.quality))

    print(f"Total images: {len(work)}")

    created = 0
    repaired = 0
    already_cached = 0
    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_resize_one, w): w for w in work}
        with tqdm(total=len(work), unit="img", desc="caching") as pbar:
            for future in as_completed(futures):
                status, _ = future.result()
                if status == "created":
                    created += 1
                elif status == "repaired":
                    repaired += 1
                elif status == "exists":
                    already_cached += 1
                else:
                    errors += 1
                pbar.update(1)

    print(
        f"Done: {created} cached, {repaired} repaired, "
        f"{already_cached} already cached, {errors} errors"
    )
    cache_size_gb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / (1024**3)
    print(f"Cache size: {cache_size_gb:.1f} GB")


if __name__ == "__main__":
    main()
