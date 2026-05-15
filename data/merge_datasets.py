"""
Merge two Kaggle datasets into a single YOLO-format dataset.

Supports two input formats:
  - YOLO (.txt labels + images)
  - COCO JSON (annotations.json + images)

Usage:
  python data/merge_datasets.py \
    --datasets data/raw/dataset1 data/raw/dataset2 \
    --output   data/merged \
    --val-ratio 0.15 \
    --test-ratio 0.05

Each dataset folder should follow one of these layouts:
  YOLO layout:
    dataset/
      images/   (or train/images, valid/images)
      labels/   (or train/labels, valid/labels)

  COCO layout:
    dataset/
      images/
      annotations/instances_*.json  (or _annotations.coco.json)
"""

import argparse
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# COCO → YOLO conversion
# ---------------------------------------------------------------------------

def coco_to_yolo(ann_path: Path, images_dir: Path, out_images: Path, out_labels: Path, class_map: dict):
    """Convert a COCO JSON annotation file to YOLO .txt files."""
    with open(ann_path) as f:
        coco = json.load(f)

    # Build id → filename map
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

    # Build category id → class index map (sorted by id for reproducibility)
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_to_idx = {}
    for cat in cats:
        name = cat["name"]
        if name not in class_map:
            class_map[name] = len(class_map)
        cat_to_idx[cat["id"]] = class_map[name]

    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    copied = 0
    for img_id, anns in anns_by_image.items():
        fname = id_to_file[img_id]
        w, h = id_to_size[img_id]

        src_img = images_dir / fname
        if not src_img.exists():
            src_img = images_dir / Path(fname).name
        if not src_img.exists():
            print(f"  [warn] image not found: {fname}")
            continue

        stem = Path(fname).stem
        dst_img = out_images / src_img.name
        shutil.copy2(src_img, dst_img)

        lines = []
        for ann in anns:
            cls = cat_to_idx[ann["category_id"]]
            x, y, bw, bh = ann["bbox"]   # COCO: top-left x,y + width,height
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            bw /= w
            bh /= h
            cx, cy, bw, bh = [max(0.0, min(1.0, v)) for v in (cx, cy, bw, bh)]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (out_labels / f"{stem}.txt").write_text("\n".join(lines))
        copied += 1

    return copied


# ---------------------------------------------------------------------------
# YOLO dataset copy
# ---------------------------------------------------------------------------

def find_yolo_splits(root: Path):
    """Return list of (images_dir, labels_dir) pairs found in a YOLO dataset."""
    pairs = []
    # Standard flat layout
    if (root / "images").exists() and (root / "labels").exists():
        pairs.append((root / "images", root / "labels"))
    # Split layout: train/, valid/, test/ each with images/ and labels/
    for split in ("train", "valid", "val", "test"):
        idir = root / split / "images"
        ldir = root / split / "labels"
        if idir.exists() and ldir.exists():
            pairs.append((idir, ldir))
    return pairs


def copy_yolo_split(images_dir: Path, labels_dir: Path, out_images: Path, out_labels: Path, class_map: dict, dataset_name: str):
    """Copy YOLO images+labels, remapping class indices via class_map."""
    # First pass: collect all class names from label files to build the map
    label_files = list(labels_dir.glob("*.txt"))
    # We need the dataset's own names.txt or data.yaml to know the mapping
    names = _load_yolo_names(labels_dir.parent.parent) or _load_yolo_names(labels_dir.parent)

    copied = 0
    for lf in label_files:
        img_file = None
        for ext in IMG_EXTS:
            candidate = images_dir / (lf.stem + ext)
            if candidate.exists():
                img_file = candidate
                break
        if img_file is None:
            print(f"  [warn] no image for label: {lf.name}")
            continue

        lines = lf.read_text().strip().splitlines()
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            old_idx = int(parts[0])
            if names and old_idx < len(names):
                name = names[old_idx]
                if name not in class_map:
                    class_map[name] = len(class_map)
                new_idx = class_map[name]
            else:
                # No name map available — keep index as-is, prefix with dataset
                key = f"{dataset_name}_{old_idx}"
                if key not in class_map:
                    class_map[key] = len(class_map)
                new_idx = class_map[key]
            new_lines.append(f"{new_idx} {' '.join(parts[1:])}")

        dst_stem = f"{dataset_name}_{lf.stem}"
        shutil.copy2(img_file, out_images / (dst_stem + img_file.suffix))
        (out_labels / f"{dst_stem}.txt").write_text("\n".join(new_lines))
        copied += 1

    return copied


def _load_yolo_names(root: Path):
    """Try to load class names from data.yaml or obj.names in root."""
    for yaml_name in ("data.yaml", "dataset.yaml", "obj.data"):
        p = root / yaml_name
        if p.exists():
            import yaml
            try:
                data = yaml.safe_load(p.read_text())
                if isinstance(data, dict):
                    names = data.get("names")
                    if isinstance(names, list):
                        return names
                    if isinstance(names, dict):
                        return [names[i] for i in sorted(names)]
            except Exception:
                pass
    for names_file in root.glob("*.names"):
        return names_file.read_text().strip().splitlines()
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Merge YOLO/COCO datasets into one YOLO dataset.")
    ap.add_argument("--datasets", nargs="+", required=True, help="Paths to raw dataset folders")
    ap.add_argument("--output", default="data/merged", help="Output merged dataset path")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.output)

    # Staging area — collect everything before splitting
    stage_images = out / "_stage" / "images"
    stage_labels = out / "_stage" / "labels"
    stage_images.mkdir(parents=True, exist_ok=True)
    stage_labels.mkdir(parents=True, exist_ok=True)

    class_map: dict[str, int] = {}
    total = 0

    for ds_path_str in args.datasets:
        ds = Path(ds_path_str)
        if not ds.exists():
            print(f"[error] Dataset not found: {ds}")
            continue

        print(f"\n[dataset] {ds.name}")
        ds_name = ds.name.replace(" ", "_")

        # Detect COCO JSON
        coco_jsons = list(ds.glob("**/*.json"))
        coco_jsons = [j for j in coco_jsons if "annotation" in j.name.lower() or "coco" in j.name.lower()]

        if coco_jsons:
            for cj in coco_jsons:
                images_dir = cj.parent.parent / "images"
                if not images_dir.exists():
                    images_dir = cj.parent / "images"
                print(f"  COCO JSON: {cj.name}")
                n = coco_to_yolo(cj, images_dir, stage_images, stage_labels, class_map)
                print(f"  -> {n} samples converted")
                total += n
        else:
            pairs = find_yolo_splits(ds)
            if not pairs:
                print(f"  [warn] No recognised layout in {ds}")
                continue
            for img_dir, lbl_dir in pairs:
                print(f"  YOLO split: {img_dir.relative_to(ds)}")
                n = copy_yolo_split(img_dir, lbl_dir, stage_images, stage_labels, class_map, ds_name)
                print(f"  -> {n} samples copied")
                total += n

    if total == 0:
        print("\n[error] No samples collected. Check your dataset paths.")
        return

    print(f"\nTotal samples staged: {total}")
    print(f"Classes discovered: {class_map}")

    # Split into train / val / test
    all_stems = [p.stem for p in stage_labels.glob("*.txt")]
    random.shuffle(all_stems)

    n_test = int(len(all_stems) * args.test_ratio)
    n_val  = int(len(all_stems) * args.val_ratio)
    test_set  = set(all_stems[:n_test])
    val_set   = set(all_stems[n_test:n_test + n_val])
    train_set = set(all_stems[n_test + n_val:])

    split_map = {s: "train" for s in train_set}
    split_map.update({s: "val" for s in val_set})
    split_map.update({s: "test" for s in test_set})

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for lf in stage_labels.glob("*.txt"):
        split = split_map[lf.stem]
        shutil.move(str(lf), out / "labels" / split / lf.name)

    for img in stage_images.iterdir():
        split = split_map.get(img.stem)
        if split:
            shutil.move(str(img), out / "images" / split / img.name)

    shutil.rmtree(out / "_stage")

    # Write dataset.yaml
    names_list = [k for k, _ in sorted(class_map.items(), key=lambda x: x[1])]
    yaml_content = (
        f"path: {out.resolve()}\n\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: {len(names_list)}\n\n"
        f"names:\n"
    )
    for i, name in enumerate(names_list):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nDataset YAML written to: {yaml_path}")

    print(f"\nSplit summary:")
    print(f"  train : {len(train_set)}")
    print(f"  val   : {len(val_set)}")
    print(f"  test  : {len(test_set)}")
    print("\nDone. Next step: python train.py --data data/merged/dataset.yaml")


if __name__ == "__main__":
    main()
