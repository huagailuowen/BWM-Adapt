#!/usr/bin/env python3
"""Build a non-executing train/infer/evaluation plan for one method matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan_video_action.methods.common.budget import write_json_atomic
from wan_video_action.methods.common.matrix import load_method_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_method_matrix(args.matrix).as_dict()
    if args.output:
        write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
