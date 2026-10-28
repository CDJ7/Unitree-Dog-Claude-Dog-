"""Small helpers to edit the merged MJCF scene before it is compiled.

The simulator's NativeSceneBuilder writes a temporary scene.xml (house + Go2)
for every session; we only ever edit that copy, never the simulator's files.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

OFFSCREEN_SIZE = (1920, 1080)     # evidence close-ups and dataset renders


def load(path):
    tree = ET.parse(path)
    return tree, tree.getroot()


def save(tree, path):
    tree.write(path, encoding="utf-8", xml_declaration=True)


def child(parent, tag):
    """First child with this tag, created if missing."""
    el = parent.find(tag)
    if el is None:
        el = ET.SubElement(parent, tag)
    return el


def set_offscreen(root, width=OFFSCREEN_SIZE[0], height=OFFSCREEN_SIZE[1]):
    glob = child(child(root, "visual"), "global")
    glob.set("offwidth", str(width))
    glob.set("offheight", str(height))


def append_xml(root, section, fragment):
    """Append the elements of an XML fragment (string) to <section> (worldbody, asset, ...)."""
    holder = ET.fromstring(f"<x>{fragment}</x>")
    sec = child(root, section)
    for el in list(holder):
        sec.append(el)


def patch_file(path, fn):
    """Load scene.xml, call fn(root), save it back."""
    tree, root = load(path)
    fn(root)
    save(tree, path)
    return Path(path)


def base_patch(path):
    """Applied to every session: allow 1920x1080 offscreen renders."""
    patch_file(path, set_offscreen)
