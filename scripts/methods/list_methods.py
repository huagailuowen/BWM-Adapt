#!/usr/bin/env python3
"""List registered method protocol definitions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan_video_action.methods import list_method_specs


def main() -> None:
    print(json.dumps(
        [spec.as_dict() for spec in list_method_specs()],
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
