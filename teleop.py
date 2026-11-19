#!/usr/bin/env python
"""Keyboard teleoperation (shortcut for `python launch.py --commander teleop`).

  W / S insert / retract   <- / -> bend CCW / CW   Shift fast   Space straighten
  Enter record   R reset   T next target   N / P vessel   F feature map   Esc quit

All launch.py options work, e.g.  python teleop.py --model vessel_03 --record
"""
import sys

from launch import main

if __name__ == "__main__":
    main(["--commander", "teleop", *sys.argv[1:]])
