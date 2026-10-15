#!/usr/bin/env python3
"""Download a few real sample photos into sim/photos/ for use as photo-textured billboards in the sim
(YOLO is trained on real photos and often ignores plain coloured shapes). Then add them under `billboards:`
in config/sim.yaml. Photos are git-ignored. Use only images you are allowed to use."""
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
}

os.makedirs(os.path.join(HERE, "photos"), exist_ok=True)
for name, url in PHOTOS.items():
    dest = os.path.join(HERE, "photos", name)
    if os.path.exists(dest):
        print("have", name)
        continue
    try:
        urllib.request.urlretrieve(url, dest)
        print("downloaded", name)
    except Exception as e:
        print("FAILED", name, e, "- download any JPG into sim/photos/ yourself")
print("Now uncomment/add `billboards:` entries in config/sim.yaml (see the example there).")
