"""Decide which detections are trustworthy enough to add to the map."""


def _inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


def usable_observations(dets, image_shape, occlusion_overlap_frac=0.25, border_margin_px=3):
    """Drop detections whose depth is likely wrong:

    * truncated by the image border (the box centre is not the object centre),
    * mostly hidden behind another detection. Floor-standing objects that are nearer have their box bottom lower
      in the image, so a box is "occluded" when another box with a lower bottom edge covers more than
      `occlusion_overlap_frac` of it. (Depth can't be used for this: the occluded box's depth already picked up the
      occluder.)
    Detections without a 3D position are skipped.
    """
    h, w = image_shape[:2]
    cand = [d for d in dets if d.xyz_world is not None and d.distance_m is not None]
    keep = []
    for d in cand:
        x1, y1, x2, y2 = d.bbox
        if x1 <= border_margin_px or y1 <= border_margin_px or x2 >= w - 1 - border_margin_px or y2 >= h - 1 - border_margin_px:
            continue
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        occluded = any(o is not d and o.bbox[3] > y2 and _inter(d.bbox, o.bbox) / area > occlusion_overlap_frac for o in dets)
        if not occluded:
            keep.append(d)
    return keep
