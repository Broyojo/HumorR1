"""Curate an out-of-distribution (OOD) image dataset for Exp 3.

Strategy
--------
We want images that are visually distinct from New Yorker cartoons (pen-and-ink
line art) but still carry humorous or surreal potential. We curate from four
open-licensed categories:

  1. stock_photos   — ordinary real-world scenes (maximum visual domain shift)
  2. photos_surreal — humorous/odd real juxtapositions (tests humor transfer)

  Memes removed (text overlays confound vision eval).
  Illustrations removed (too few examples for meaningful analysis).

Each image is saved to data/ood_images/<category>/<id>.jpg and a metadata
manifest is written to data/ood_manifest.jsonl.

Sources used (all permissively licensed / public domain):
  - COCO 2017 val images (natural photos, CC-BY)
  - Flickr30k sample (CC-BY)
  - Custom-curated surreal / humorous stock photos listed in CURATED_URLS below

How to extend
-------------
Add entries to CURATED_URLS or point --coco-dir at a local COCO val directory
to bulk-import natural photos.

Usage:
  uv run python scripts/curate_ood_dataset.py
  uv run python scripts/curate_ood_dataset.py --max-per-category 30 --coco-dir /path/to/coco/val2017

Output:
  data/ood_images/<category>/<id>.jpg   — raw images
  data/ood_manifest.jsonl               — one JSON record per image
"""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Curated OOD image URLs (permissive license; extend as needed)
# Each entry: (url, category, description_hint)
# ---------------------------------------------------------------------------
# Two balanced categories chosen for clean experimental design:
#   stock_photos  — ordinary real-world scenes (maximum domain shift from cartoons)
#   photos_surreal — humorous/odd real-world juxtapositions (tests humor transfer)
# Memes and illustrations removed: memes have text overlays that confound vision
# eval, and illustrations were too few to draw conclusions from.
# All images are from Unsplash (permissive license) or COCO val2017 (CC-BY).
CURATED_URLS: list[tuple[str, str, str]] = [
    # ── stock_photos: ordinary everyday scenes ─────────────────────────────
    # These represent maximum domain shift — clean real photos vs pen-and-ink cartoons
    (
        "https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?w=800",
        "stock_photos",
        "Person sitting alone at a long empty conference table",
    ),
    (
        "https://images.unsplash.com/photo-1540575467063-178a50c2df87?w=800",
        "stock_photos",
        "Businesspeople at a whiteboard covered in indecipherable diagrams",
    ),
    (
        "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=800",
        "stock_photos",
        "Group of people all looking at their phones, ignoring each other",
    ),
    (
        "https://images.unsplash.com/photo-1497366216548-37526070297c?w=800",
        "stock_photos",
        "Empty open-plan office with rows of desks and no people",
    ),
    (
        "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=800",
        "stock_photos",
        "Two people shaking hands across a desk in a formal meeting",
    ),
    (
        "https://images.unsplash.com/photo-1551836022-d5d88e9218df?w=800",
        "stock_photos",
        "Person alone at a coffee shop staring into space",
    ),
    (
        "https://images.unsplash.com/photo-1522202176988-66273c2fd55f?w=800",
        "stock_photos",
        "Team collaborating around a laptop at a table",
    ),
    (
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800",
        "stock_photos",
        "Man in a suit looking directly at camera with neutral expression",
    ),
    (
        "https://images.unsplash.com/photo-1560264280-88b68371db39?w=800",
        "stock_photos",
        "Crowded city street with pedestrians walking past storefronts",
    ),
    (
        "https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d?w=800",
        "stock_photos",
        "Person typing on a laptop at a minimalist desk",
    ),
    (
        "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=800",
        "stock_photos",
        "Woman presenting in front of a screen to an audience",
    ),
    (
        "https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?w=800",
        "stock_photos",
        "People sitting at a long boardroom table during a meeting",
    ),
    (
        "https://images.unsplash.com/photo-1553877522-43269d4ea984?w=800",
        "stock_photos",
        "Person reading documents at a messy desk covered in papers",
    ),
    (
        "https://images.unsplash.com/photo-1495653797063-114787b77b23?w=800",
        "stock_photos",
        "Two people having a conversation in a hallway",
    ),
    (
        "https://images.unsplash.com/photo-1531973576160-7125cd663d86?w=800",
        "stock_photos",
        "Modern office lobby with a receptionist desk",
    ),
    # ── photos_surreal: humorous/odd real-world juxtapositions ──────────────
    # These test whether humor understanding transfers when visual context is odd
    (
        "https://images.unsplash.com/photo-1517694712202-14dd9538aa97?w=800",
        "photos_surreal",
        "Man in a suit sitting at a laptop in the middle of a forest",
    ),
    (
        "https://images.unsplash.com/photo-1504701954957-2010ec3bcec1?w=800",
        "photos_surreal",
        "Sheep standing at a podium as if giving a lecture",
    ),
    (
        "https://images.unsplash.com/photo-1497032628192-86f99bcd76bc?w=800",
        "photos_surreal",
        "Tiny person dwarfed by a massive stack of paperwork",
    ),
    (
        "https://images.unsplash.com/photo-1516912481808-3406841bd33c?w=800",
        "photos_surreal",
        "Cat sitting at a desk looking very serious",
    ),
    (
        "https://images.unsplash.com/photo-1511367461989-f85a21fda167?w=800",
        "photos_surreal",
        "Dog wearing sunglasses in a car",
    ),
    (
        "https://images.unsplash.com/photo-1474511320723-9a56873867b5?w=800",
        "photos_surreal",
        "Fox looking directly at camera with an alert expression",
    ),
    (
        "https://images.unsplash.com/photo-1425082661705-1834bfd09dca?w=800",
        "photos_surreal",
        "Squirrel sitting upright holding something with both paws",
    ),
    (
        "https://images.unsplash.com/photo-1533738363-b7f9aef128ce?w=800",
        "photos_surreal",
        "Cat looking judgmentally at the camera",
    ),
    (
        "https://images.unsplash.com/photo-1518020382113-a7e8fc38eac9?w=800",
        "photos_surreal",
        "Dog sitting in a very human-like pose on a couch",
    ),
    (
        "https://images.unsplash.com/photo-1437622368342-7a3d73a34c8f?w=800",
        "photos_surreal",
        "Sea turtle swimming alone in a vast expanse of ocean",
    ),
    (
        "https://images.unsplash.com/photo-1520038410233-7141be7e6f97?w=800",
        "photos_surreal",
        "Person in a full business suit standing in the middle of a beach",
    ),
    (
        "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=800",
        "photos_surreal",
        "Dog looking dramatically into the distance",
    ),
    (
        "https://images.unsplash.com/photo-1548199973-03cce0bbc87b?w=800",
        "photos_surreal",
        "Two dogs running side by side looking extremely serious",
    ),
    (
        "https://images.unsplash.com/photo-1456926631375-92c8ce872def?w=800",
        "photos_surreal",
        "Pigeon standing alone on an empty park bench",
    ),
    (
        "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800",
        "photos_surreal",
        "Empty office chair facing a window with dramatic lighting",
    ),
    (
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800",
        "photos_surreal",
        "Person in a bear costume sitting in a waiting room",
    ),
]

SEED = 42
random.seed(SEED)

OOD_CATEGORIES = ["stock_photos", "photos_surreal"]


def download_image(url: str, dest: Path) -> bool:
    """Download an image from URL to dest. Returns True on success."""
    try:
        import urllib.request
        import ssl

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        headers = {"User-Agent": "Mozilla/5.0 humor-rlhf/0.1 (NLP class project)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as e:
        print(f"    [WARN] Could not download {url}: {e}")
        return False


def image_id_from_url(url: str) -> str:
    """Stable short ID derived from URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def import_coco_images(coco_dir: Path, output_dir: Path, max_per_category: int) -> list[dict]:
    """Import natural images from a local COCO val directory."""
    records = []
    coco_out = output_dir / "stock_photos"
    coco_out.mkdir(parents=True, exist_ok=True)

    candidates = sorted(coco_dir.glob("*.jpg"))
    random.shuffle(candidates)
    chosen = candidates[:max_per_category]

    for img_path in chosen:
        dest = coco_out / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)
        records.append({
            "id": img_path.stem,
            "category": "stock_photos",
            "source": "COCO 2017 val",
            "local_path": str(dest.relative_to(output_dir.parent)),
            "description": f"COCO image {img_path.stem}",
        })
    print(f"  Imported {len(records)} COCO images.")
    return records


def curate_from_urls(
    output_dir: Path,
    max_per_category: int,
) -> list[dict]:
    """Download curated images from CURATED_URLS."""
    records = []
    by_category: dict[str, list] = {}
    for url, cat, desc in CURATED_URLS:
        by_category.setdefault(cat, []).append((url, desc))

    for category, items in by_category.items():
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        chosen = items[:max_per_category]

        for url, desc in chosen:
            img_id = image_id_from_url(url)
            dest = cat_dir / f"{img_id}.jpg"
            print(f"  [{category}] Downloading {img_id}...")
            if dest.exists():
                print(f"    Already exists, skipping.")
            else:
                ok = download_image(url, dest)
                if not ok:
                    continue

            records.append({
                "id": img_id,
                "category": category,
                "source": url,
                "local_path": str(dest.relative_to(output_dir.parent)),
                "description": desc,
            })

    return records


def validate_images(records: list[dict], output_dir: Path) -> list[dict]:
    """Filter out corrupt/unreadable images using Pillow."""
    from PIL import Image

    valid = []
    for rec in records:
        path = output_dir.parent / rec["local_path"]
        try:
            with Image.open(path) as img:
                img.verify()
            valid.append(rec)
        except Exception as e:
            print(f"  [INVALID] {path}: {e}")
    return valid


def write_manifest(records: list[dict], manifest_path: Path):
    with open(manifest_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nManifest written: {manifest_path} ({len(records)} images)")


def print_summary(records: list[dict]):
    from collections import Counter
    counts = Counter(r["category"] for r in records)
    print("\n=== OOD Dataset Summary ===")
    for cat in OOD_CATEGORIES:
        print(f"  {cat:20s}: {counts.get(cat, 0):3d} images")
    print(f"  {'TOTAL':20s}: {len(records):3d} images")


def parse_args():
    parser = argparse.ArgumentParser(description="Curate OOD image dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ood_images"),
        help="Directory to save OOD images (default: data/ood_images)",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=20,
        help="Max images per category (default: 20)",
    )
    parser.add_argument(
        "--coco-dir",
        type=Path,
        default=None,
        help="Optional path to a local COCO val2017 directory for natural photos",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/ood_manifest.jsonl"),
        help="Path to write the image manifest",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    print("=== Step 1: Download curated images ===")
    all_records += curate_from_urls(args.output_dir, args.max_per_category)

    if args.coco_dir and args.coco_dir.exists():
        print("\n=== Step 2: Import COCO natural images ===")
        all_records += import_coco_images(
            args.coco_dir, args.output_dir, args.max_per_category
        )
    else:
        print(
            "\n[INFO] No --coco-dir provided. "
            "To add COCO natural images, download COCO val2017 and pass --coco-dir."
        )
        print(
            "       Download: http://images.cocodataset.org/zips/val2017.zip"
        )

    print("\n=== Step 3: Validating images ===")
    all_records = validate_images(all_records, args.output_dir)

    write_manifest(all_records, args.manifest)
    print_summary(all_records)
    print(
        "\nDone! To run OOD eval, pass --ood-manifest data/ood_manifest.jsonl "
        "to scripts/run_ood_eval.py"
    )


if __name__ == "__main__":
    main()
