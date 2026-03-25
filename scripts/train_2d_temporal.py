#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

from filament_train import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["tifs2d", "--mode", "2d"])
    elif "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "2d"])
    main()
