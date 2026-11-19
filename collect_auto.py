#!/usr/bin/env python
"""Generate demonstrations with the scripted autopilot, headless, same format as teleop
(shortcut for `python launch.py --commander autopilot --headless --record ...`).

  python collect_auto.py                                   # every target of every model, once
  python collect_auto.py --models vessel_02,vessel_03 --episodes 5 --noise 0.3
"""
import sys

from launch import main

if __name__ == "__main__":
    main(["--commander", "autopilot", "--headless", "--record", *sys.argv[1:]])
