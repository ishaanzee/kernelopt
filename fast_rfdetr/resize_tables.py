"""Exact per-column / per-row interpolation tables reproducing cv2.resize(img, (W, H)) for uint8 BGR, INTER_LINEAR,
as done by the opencv-python 4.14 macOS arm64 wheel (Ballform's cv2).

That wheel routes the resize through one of two implementations:

* KleidiCV 26.03 HAL (Arm), used when both axes strictly downscale and the horizontal ratio is at most 3
  (both Ballform inputs: 1920x1080 and 1152x1080 -> 640x640). 16.16 fixed-point coordinates, 8-bit fractions,
  vertical lerp then horizontal lerp, each rounded as NEON vraddhn: ((a<<8) + (b-a)*f + 128) >> 8.
  Source x coordinates are generated per 16-lane vector with incremental stepping, which is replicated here
  (kleidicv/src/resize/resize_linear_generic_u8_neon.h).
* OpenCV's own fixed-point path otherwise (imgproc/src/resize.cpp): 11-bit weights, horizontal pass in int32,
  vertical pass with the NEON VResizeLinearVec_32s8u rounding.

Both are expressed as the same 4-tap form so a single GPU kernel can evaluate either:
    mode 0 (KleidiCV):  left  = (A*256 + (C-A)*fy + 128) >> 8,  right = (B*256 + (D-B)*fy + 128) >> 8,
                        out   = (left*256 + (right-left)*fx + 128) >> 8
    mode 1 (OpenCV):    h0 = A*a0 + B*a1,  h1 = C*a0 + D*a1,
                        out = sat_u8((((h0>>4)*b0 >> 16) + ((h1>>4)*b1 >> 16) + 2) >> 2)
with A/B the top-row taps and C/D the bottom-row taps. Tables are indexed per output *element* (x*3 + c) so the
KleidiCV per-lane quirks are captured exactly.
"""
from __future__ import annotations

import functools

import numpy as np

FIXP = 16
HALF = 1 << (FIXP - 1)
STEP, HALF_STEP, CH = 16, 8, 3


def _rdiv(nom: int, den: int) -> int:
    return (nom + den // 2) // den


def _aligned(x: int, nom: int, den: int) -> int:
    return _rdiv(((x << FIXP) + HALF) * nom, den) - HALF + (1 << (FIXP - 9))


def uses_kleidicv(src_w, src_h, dst_w, dst_h) -> bool:
    return (dst_w * 3 >= src_w and dst_w < src_w and dst_h < src_h
            and dst_w * CH >= 8 and src_w * CH >= 32)


def _kleidi_x(src_w: int, dst_w: int):
    """Returns per output element: source element index of the left tap, 8-bit x fraction."""
    ratio = 2 if src_w / dst_w < 1.8 else 3
    n_el = dst_w * CH
    src_el = np.zeros(n_el, np.int64)
    xfrac = np.zeros(n_el, np.int64)
    scale_x = lambda k: _rdiv((k * src_w) << FIXP, dst_w)  # noqa: E731
    to_src_x = lambda dx: _aligned(dx, src_w, dst_w)  # noqa: E731
    one = _rdiv(src_w << FIXP, dst_w)

    two_x = (n_el // (2 * STEP)) if src_w * CH >= STEP * ratio else 0
    n_full = two_x * 2
    remaining = n_el - two_x * 2 * STEP
    n_half = -(-remaining // HALF_STEP)

    # lane patterns (pixel offset, channel) for vectors starting on channel 0 / 1 / 2
    pix = {0: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5],
           1: [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
           2: [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]}
    idx_diff = {0: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
                1: [0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1],
                2: [0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]}

    def fill_vector(dst_e, sx, ip):
        base_int = sx >> FIXP
        f0 = sx & 0xFFFF
        for lane in range(STEP):
            pos = f0 + scale_x(pix[ip][lane])
            lane_int = (pos >> FIXP) & 0xFF
            # vmulq_u8 by 3, vqsubq_u8 of the in-pixel index, then add the lane's channel offset
            idx = max(((lane_int * 3) & 0xFF) - ip, 0) + idx_diff[ip][lane]
            src_el[dst_e + lane] = base_int * CH + ip + idx
            xfrac[dst_e + lane] = (pos >> 8) & 0xFF

    def fill_scalar(dst_e, n, ip, sx):
        """fill_idx_xfrac: first pixel from sx, then += one per pixel. Returns element indices (absolute)."""
        j = 0
        s_el = (sx >> FIXP) * CH + ip
        xf = (sx & 0xFFFF) >> 8
        while j < CH - ip and j < n:
            src_el[dst_e + j], xfrac[dst_e + j] = s_el + j, xf
            j += 1
        while j < n:
            sx += one
            s_el = (sx >> FIXP) * CH
            xf = (sx & 0xFFFF) >> 8
            k = 0
            while j < n and k < CH:
                src_el[dst_e + j], xfrac[dst_e + j] = s_el + k, xf
                j += 1
                k += 1

    def needs_pullback(dst_idx):
        ip = dst_idx % CH
        src_idx = (to_src_x(dst_idx // CH) >> FIXP) * CH + ip
        return src_idx + STEP * ratio > src_w * CH

    dst_e = handled = 0
    if n_full > 0:
        if n_full > 3:
            cand = n_full - 1
            while True:
                if not needs_pullback(cand * STEP):
                    break
                cand -= 1
                if cand <= 0:
                    break
            n_wout = 0 if (cand == 0 and needs_pullback(0)) else cand + 1
            sx = to_src_x(0)
            cnt = 0
            five, six = _rdiv((src_w * 5) << FIXP, dst_w), _rdiv((src_w * 6) << FIXP, dst_w)
            for _ in range(n_wout // 3):
                if cnt == 170:
                    sx, cnt = to_src_x(dst_e // CH), 0
                else:
                    cnt += 1
                fill_vector(dst_e, sx, 0)
                sx += five
                fill_vector(dst_e + STEP, sx, 1)
                sx += five
                fill_vector(dst_e + 2 * STEP, sx, 2)
                sx += six
                handled += 3
                dst_e += 3 * STEP
        while handled < n_full:
            fill_scalar(dst_e, STEP, dst_e % CH, to_src_x(dst_e // CH))
            handled += 1
            dst_e += STEP
    for _ in range(n_half):
        dst_e = min(dst_e, n_el - HALF_STEP)
        fill_scalar(dst_e, HALF_STEP, dst_e % CH, to_src_x(dst_e // CH))
        dst_e += HALF_STEP
    return src_el, xfrac


def _kleidi_y(src_h: int, dst_h: int):
    sy = np.array([_aligned(d, src_h, dst_h) for d in range(dst_h)], dtype=np.int64)
    row = sy >> FIXP
    yfrac = (sy - (row << FIXP)) >> (FIXP - 8)
    return row, np.minimum(row + 1, src_h - 1), yfrac


def _opencv_axis(src_len: int, dst_len: int, clamp_coord: bool):
    """x: out-of-range coordinates are clamped (frac forced to 0). y: only the row indices are clipped and the
    fractional weights are kept, so both taps can read the same border row (resizeGeneric_ yofs/ibeta)."""
    scale = src_len / dst_len
    f = ((np.arange(dst_len, dtype=np.float64) + 0.5) * scale - 0.5).astype(np.float32)
    s = np.floor(f).astype(np.int64)
    frac = (f - s.astype(np.float32)).astype(np.float32)
    if clamp_coord:
        low = s < 0
        frac[low], s[low] = 0, 0
        high = s >= src_len - 1
        frac[high], s[high] = 0, src_len - 1
    w1 = np.rint(frac * np.float32(2048)).astype(np.int64)
    w0 = np.rint((np.float32(1) - frac) * np.float32(2048)).astype(np.int64)
    return np.clip(s, 0, src_len - 1), np.clip(s + 1, 0, src_len - 1), w0, w1


@functools.lru_cache(maxsize=16)
def tables(src_w: int, src_h: int, dst_w: int = 640, dst_h: int = 640):
    """Returns (mode, xtab int32 [dst_w*3, 4], ytab int32 [dst_h, 4]).

    xtab columns: left element index, right element index, weight a (fx or a0), weight b (unused or a1)
    ytab columns: top row, bottom row, weight (fy or b0), weight (unused or b1)"""
    if uses_kleidicv(src_w, src_h, dst_w, dst_h):
        el, fx = _kleidi_x(src_w, dst_w)
        r0, r1, fy = _kleidi_y(src_h, dst_h)
        xt = np.stack([el, el + CH, fx, np.zeros_like(fx)], 1)
        yt = np.stack([r0, r1, fy, np.zeros_like(fy)], 1)
        mode = 0
    else:
        x0, x1, a0, a1 = _opencv_axis(src_w, dst_w, clamp_coord=True)
        y0, y1, b0, b1 = _opencv_axis(src_h, dst_h, clamp_coord=False)
        c = np.tile(np.arange(CH), dst_w)
        xt = np.stack([np.repeat(x0, CH) * CH + c, np.repeat(x1, CH) * CH + c, np.repeat(a0, CH), np.repeat(a1, CH)], 1)
        yt = np.stack([y0, y1, b0, b1], 1)
        mode = 1
    return mode, np.ascontiguousarray(xt, np.int32), np.ascontiguousarray(yt, np.int32)


def resize_reference(img: np.ndarray, dst_w: int = 640, dst_h: int = 640) -> np.ndarray:
    """Numpy evaluation of the tables (bit-exact model of cv2.resize for uint8 3-channel)."""
    h, w, _ = img.shape
    mode, xt, yt = tables(w, h, dst_w, dst_h)
    flat = img.reshape(h, w * CH).astype(np.int64)
    top, bot = flat[yt[:, 0]], flat[yt[:, 1]]                 # [dst_h, w*3]
    A, B = top[:, xt[:, 0]], top[:, xt[:, 1]]
    C, D = bot[:, xt[:, 0]], bot[:, xt[:, 1]]
    if mode == 0:
        fy, fx = yt[:, 2:3].astype(np.int64), xt[None, :, 2].astype(np.int64)
        left = (A * 256 + (C - A) * fy + 128) >> 8
        right = (B * 256 + (D - B) * fy + 128) >> 8
        out = (left * 256 + (right - left) * fx + 128) >> 8
    else:
        a0, a1 = xt[None, :, 2].astype(np.int64), xt[None, :, 3].astype(np.int64)
        b0, b1 = yt[:, 2:3].astype(np.int64), yt[:, 3:4].astype(np.int64)
        h0, h1 = A * a0 + B * a1, C * a0 + D * a1
        out = ((((h0 >> 4) * b0) >> 16) + (((h1 >> 4) * b1) >> 16) + 2) >> 2
    return np.clip(out, 0, 255).astype(np.uint8).reshape(dst_h, dst_w, CH)

