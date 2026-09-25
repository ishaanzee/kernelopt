"""Accuracy check against the golden ONNX Runtime CPU fp32 outputs (Ballform decode + object matching)."""
from __future__ import annotations

import numpy as np

KEPT_CLASSES = {1, 2, 4, 5, 6, 7, 8, 9, 10}


def decode(boxes, logits, threshold=.25):
    """Copy of ballform app.basketball.decode for one image: [(class, score, xyxy)]."""
    boxes, logits = np.asarray(boxes).reshape(300, 4), np.asarray(logits).reshape(300, 11)
    scores = (1 / (1 + np.exp(-np.clip(logits, -80, 80)))).ravel()
    objects = []
    for flat_index in np.argsort(scores)[-300:][::-1]:
        score = float(scores[flat_index])
        if score < threshold:
            break
        query, category = divmod(int(flat_index), 11)
        if category not in KEPT_CLASSES:
            continue
        cx, cy, w, h = boxes[query].astype(np.float64)
        objects.append((category, score, tuple(np.clip([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 0, 1))))
    return objects


def load_reference(npz):
    """Returns list of (name, image_bgr, golden_boxes[300,4], golden_logits[300,11]) for all 11 images."""
    z = np.load(npz)
    items = [(f"frame{i}", z["frames_bgr"][i], z["frame_boxes"][i], z["frame_logits"][i])
             for i in range(len(z["frames_bgr"]))]
    items += [(f"crop{i}", z["crops_bgr"][i], z["crop_boxes"][i], z["crop_logits"][i])
              for i in range(len(z["crops_bgr"]))]
    return items


def match_objects(golden, candidate, min_conf=.4, conf_tol=.03, box_tol=.01):
    """Greedy one-to-one match of golden objects (conf >= min_conf) to candidates of the same class."""
    used, misses, worst_conf, worst_box = set(), [], 0.0, 0.0
    for cls, conf, box in (g for g in golden if g[1] >= min_conf):
        best = None
        for j, (ccls, cconf, cbox) in enumerate(candidate):
            if j in used or ccls != cls:
                continue
            dbox = float(np.max(np.abs(np.subtract(box, cbox))))
            dconf = abs(conf - cconf)
            if dbox <= box_tol and dconf <= conf_tol and (best is None or dbox < best[1]):
                best = (j, dbox, dconf)
        if best is None:
            misses.append((cls, round(conf, 4), tuple(round(v, 4) for v in box)))
        else:
            used.add(best[0])
            worst_box, worst_conf = max(worst_box, best[1]), max(worst_conf, best[2])
    total = sum(1 for g in golden if g[1] >= min_conf)
    return total, misses, worst_conf, worst_box


def accuracy_report(run_batch, npz, batch_size=1, verbose=True):
    """run_batch(list_of_bgr) -> (boxes [N,300,4], logits [N,300,11]). Checks all 11 reference images."""
    ref = load_reference(npz)
    outs = []
    for i in range(0, len(ref), batch_size):
        chunk = ref[i:i + batch_size]
        boxes, logits = run_batch([c[1] for c in chunk])
        outs += [(np.asarray(boxes[k]), np.asarray(logits[k])) for k in range(len(chunk))]
    total = matched = 0
    max_dlogit = max_dbox = worst_conf = worst_box = 0.0
    all_misses = []
    for (name, _, gb, gl), (b, lg) in zip(ref, outs):
        max_dlogit = max(max_dlogit, float(np.abs(lg.reshape(300, 11) - gl).max()))
        max_dbox = max(max_dbox, float(np.abs(b.reshape(300, 4) - gb).max()))
        n, misses, wc, wb = match_objects(decode(gb, gl), decode(b, lg))
        total += n
        matched += n - len(misses)
        worst_conf, worst_box = max(worst_conf, wc), max(worst_box, wb)
        all_misses += [(name, m) for m in misses]
    report = dict(objects=total, matched=matched, passed=matched == total,
                  max_logit_diff=max_dlogit, max_box_diff=max_dbox,
                  worst_matched_conf_diff=worst_conf, worst_matched_box_diff=worst_box, misses=all_misses)
    if verbose:
        print(f"accuracy: {matched}/{total} objects (conf>=0.4) matched  "
              f"max|dlogit|={max_dlogit:.5f} max|dbox|={max_dbox:.5f}  "
              f"worst matched dconf={worst_conf:.5f} dbox={worst_box:.5f}  {'PASS' if report['passed'] else 'FAIL'}")
        for name, m in all_misses[:10]:
            print("  miss", name, m)
    return report
