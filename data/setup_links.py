import os
from pathlib import Path

BASE = Path(__file__).parent
RAW  = BASE / "raw/Dataset military equipment"
OUT  = BASE / "merged"

for split in ("train", "val", "test"):
    src_img = RAW / f"images_{split}"
    src_lbl = RAW / f"labels_{split}"
    dst_img = OUT / "images" / split
    dst_lbl = OUT / "labels" / split

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for f in src_img.iterdir():
        dst = dst_img / f.name
        if not dst.exists():
            os.link(f, dst)

    for f in src_lbl.iterdir():
        dst = dst_lbl / f.name
        if not dst.exists():
            os.link(f, dst)

    print(f"{split}: {len(list(dst_img.iterdir()))} images, {len(list(dst_lbl.iterdir()))} labels")

print("Done")
