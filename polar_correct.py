#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys

from PIL import Image

H = 512
W = 512
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
LATLON_RE = re.compile(r"lat_(-?\d+(?:\.\d+)?)_long_(-?\d+(?:\.\d+)?)")


def norm_lon(lon):
    lon = ((lon + 180.0) % 360.0) - 180.0
    return round(lon, 4)


def parse_key(name):
    m = LATLON_RE.search(name)
    if not m:
        return None
    return round(float(m.group(1)), 4), norm_lon(float(m.group(2)))


def build_grids(image_dir, mask_dir, labels):
    mask_by_key = {}
    if mask_dir:
        for f in os.listdir(mask_dir):
            k = parse_key(f)
            if k is not None:
                mask_by_key[k] = f

    label_by_key = {}
    label_name = {}
    if labels is not None:
        for name, craters in labels.items():
            k = parse_key(name)
            if k is not None:
                label_by_key[k] = craters
                label_name[k] = name

    image_grid, mask_grid, label_grid = {}, {}, {}
    unreadable = []
    for f in sorted(os.listdir(image_dir)):
        if not f.lower().endswith(IMG_EXTS):
            continue
        key = parse_key(f)
        if key is None:
            continue
        try:
            with Image.open(os.path.join(image_dir, f)) as im:
                im.load()
        except (OSError, SyntaxError):
            unreadable.append(f)
            continue
        image_grid[key] = f
        if key in mask_by_key:
            mask_grid[key] = mask_by_key[key]
        if key in label_by_key:
            label_grid[key] = label_by_key[key]

    if unreadable:
        print(f"[warn] {len(unreadable)} unreadable tile(s) treated as missing: {unreadable}")
    if labels is not None and not label_grid:
        print("[warn] no image tile matched a labels entry.")
        print(f"       example image file: {next(iter(image_grid.values()), None)}")
        print(f"       example label key:  {next(iter(labels), None)}")
    return image_grid, mask_grid, label_grid, label_name


def infer_grid_step(image_grid):
    lons = sorted({lon for _, lon in image_grid})
    diffs = [round(b - a, 4) for a, b in zip(lons, lons[1:]) if b - a > 1e-6]
    return min(diffs) if diffs else 2.0


def transform_labels(craters, x0, scale, keep_all):
    out = []
    for cx, cy, rx, ry in craters:
        if not keep_all and not (0.0 <= cx < W and 0.0 <= cy < H):
            continue
        ncx = x0 + cx * scale
        nrx = rx * scale
        if ncx + nrx < 0 or ncx - nrx >= W:
            continue
        out.append([ncx, cy, nrx, ry])
    return out


def make_fetcher(lat, lon, step, image_dir, mask_dir, image_grid, mask_grid, label_grid, use_labels):
    def fetch(offset):
        key = (lat, norm_lon(lon + offset * step))
        if key not in image_grid:
            return None
        img = Image.open(os.path.join(image_dir, image_grid[key])).convert("RGB")
        mask = None
        if mask_dir and key in mask_grid:
            mask = Image.open(os.path.join(mask_dir, mask_grid[key])).convert("L")
        craters = label_grid.get(key) if use_labels else None
        if mask_dir:
            valid = key in mask_grid
        elif use_labels:
            valid = key in label_grid
        else:
            valid = True
        return img, mask, valid, craters
    return fetch


def stitch(new_w, fetch, max_offset, keep_all):
    canvas_img = Image.new("RGB", (W, H), (0, 0, 0))
    canvas_mask = Image.new("L", (W, H), 0)
    canvas_valid = Image.new("L", (W, H), 0)
    canvas_labels = []
    used, missing = [], []
    scale = new_w / W

    def place(offset, x):
        got = fetch(offset)
        if got is None:
            missing.append(offset)
            return
        img, mask, valid, craters = got
        canvas_img.paste(img.resize((new_w, H), Image.BILINEAR), (x, 0))
        if mask is not None:
            canvas_mask.paste(mask.resize((new_w, H), Image.NEAREST), (x, 0))
        if valid:
            canvas_valid.paste(Image.new("L", (new_w, H), 255), (x, 0))
        if craters is not None:
            canvas_labels.extend(transform_labels(craters, x, scale, keep_all))
        used.append(offset)

    x_center = (W - new_w) // 2
    place(0, x_center)

    x, offset = x_center, -1
    while x > 0 and -offset <= max_offset:
        x -= new_w
        place(offset, x)
        offset -= 1

    x, offset = x_center + new_w, 1
    while x < W and offset <= max_offset:
        place(offset, x)
        x += new_w
        offset += 1

    return canvas_img, canvas_mask, canvas_valid, canvas_labels, used, missing


def correct_tile(lat, lon, step, args, image_grid, mask_grid, label_grid, use_labels):
    compression = math.cos(math.radians(abs(lat)))
    new_w = max(int(W * compression), 1)
    fetch = make_fetcher(lat, lon, step, args.images, args.masks, image_grid, mask_grid, label_grid, use_labels)
    max_offset = min(args.max_tiles, int(round(360.0 / step)) // 2)
    img, mask, valid, labels, used, missing = stitch(new_w, fetch, max_offset, args.keep_all_labels)
    return img, mask, valid, (labels if use_labels else None), compression, new_w, used, missing


def parse_ranges(s):
    out = []
    for part in s.split(","):
        lo, hi = part.split(":")
        out.append((float(lo), float(hi)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="directory with THEMIS tiles")
    ap.add_argument("--masks", default=None, help="directory with annotation masks")
    ap.add_argument("--labels", default=None, help="labels json: {name_with_lat_long: [[cx, cy, rx, ry], ...]}")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--lat-ranges", default=None,
                    help="comma-separated lo:hi latitude ranges, e.g. --lat-ranges=-70:-60,60:70 (default: all)")
    ap.add_argument("--require-mask", action="store_true", help="only produce canvases for tiles that have a mask")
    ap.add_argument("--keep-all-labels", action="store_true",
                    help="keep craters whose centre lies outside their source tile (may duplicate)")
    ap.add_argument("--grid-step", type=float, default=None, help="longitude step between tiles (default: inferred)")
    ap.add_argument("--max-tiles", type=int, default=64, help="max neighbours fetched per side")
    args = ap.parse_args()

    if args.masks is None and args.labels is None:
        sys.exit("Provide --masks, --labels, or both.")

    labels = None
    if args.labels:
        with open(args.labels) as fh:
            labels = json.load(fh)
    use_labels = labels is not None

    ranges = parse_ranges(args.lat_ranges) if args.lat_ranges else None

    image_grid, mask_grid, label_grid, label_name = build_grids(args.images, args.masks, labels)
    step = args.grid_step or infer_grid_step(image_grid)
    n_rows = len({lat for lat, _ in image_grid})
    print(f"{len(image_grid)} image tiles, {len(mask_grid)} with masks, {len(label_grid)} with labels, "
          f"{n_rows} latitude rows, grid step {step} deg")

    targets = [
        key for key in sorted(image_grid)
        if (ranges is None or any(lo <= key[0] <= hi for lo, hi in ranges))
        and (not args.require_mask or key in mask_grid)
    ]
    if not targets:
        sys.exit("No tiles selected: check --lat-ranges and the filename pattern lat_<lat>_long_<lon>.")
    print(f"{len(targets)} tiles selected")

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "valid"), exist_ok=True)
    if args.masks:
        os.makedirs(os.path.join(args.out, "masks"), exist_ok=True)

    manifest, out_labels = [], {}
    n_padded = 0
    for i, (lat, lon) in enumerate(targets, 1):
        img, mask, valid, lab, comp, new_w, used, missing = correct_tile(
            lat, lon, step, args, image_grid, mask_grid, label_grid, use_labels)
        stem = os.path.splitext(image_grid[(lat, lon)])[0]
        fname = stem + ".png"
        img.save(os.path.join(args.out, "images", fname))
        valid.save(os.path.join(args.out, "valid", fname))
        if args.masks:
            mask.save(os.path.join(args.out, "masks", fname))
        if lab is not None:
            out_labels[label_name.get((lat, lon), f"lat_{lat}_long_{lon}")] = lab
        if missing:
            n_padded += 1
        manifest.append({
            "file": fname, "lat": lat, "lon": lon,
            "compression": round(comp, 4), "tile_width_px": new_w,
            "neighbour_offsets_used": used,
            "neighbour_offsets_missing": missing,
            "centre_has_mask": (lat, lon) in mask_grid,
            "centre_has_labels": (lat, lon) in label_grid,
            "n_craters": len(lab) if lab is not None else None,
        })
        if i % 100 == 0 or i == len(targets):
            print(f"  {i}/{len(targets)} canvases written")

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    if use_labels:
        with open(os.path.join(args.out, "labels.json"), "w") as fh:
            json.dump(out_labels, fh)
        print(f"labels.json: {len(out_labels)} canvases, {sum(len(v) for v in out_labels.values())} craters")
    print(f"{n_padded} canvases contain black padding for missing tiles")
    print(f"Done. Outputs in {args.out}/")


if __name__ == "__main__":
    main()
