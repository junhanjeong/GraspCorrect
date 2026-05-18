#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LANGSAM_ROOT = ROOT / "external" / "langsam"
if str(LANGSAM_ROOT) not in sys.path:
    sys.path.insert(0, str(LANGSAM_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="LangSAM segmentation sidecar.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--sam-type", default="sam2.1_hiera_small")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    args = parser.parse_args()

    from lang_sam import LangSAM  # type: ignore

    model = LangSAM(sam_type=args.sam_type)
    image = Image.open(args.image).convert("RGB")
    result = model.predict(
        [image],
        [args.prompt],
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )[0]
    masks = np.asarray(result.get("masks", []))
    if masks.size == 0:
        raise SystemExit(f"No LangSAM masks for prompt: {args.prompt!r}")
    scores = np.asarray(result.get("mask_scores", result.get("scores", np.ones(len(masks)))))
    idx = int(np.argmax(scores.reshape(-1))) if scores.size else 0
    mask = masks[idx].astype(bool)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, mask)
    if args.metadata_output:
        meta = {
            "prompt": args.prompt,
            "mask_index": idx,
            "area": int(mask.sum()),
            "scores": scores.reshape(-1).astype(float).tolist() if scores.size else [],
        }
        Path(args.metadata_output).write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
