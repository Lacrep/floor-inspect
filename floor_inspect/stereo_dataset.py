# -*- coding: utf-8 -*-
"""stereo_dataset — อ่านชุดข้อมูลถนนจริงที่มี point cloud เมตริก แล้วป้อนเข้าท่อเดียวกับกล้อง D455

รองรับรูปแบบของ "stereo pothole datasets" (Rui Fan, MIT) — ดู docs/THIRD_PARTY_NOTICES.md
  rgb/NN.png       ภาพสี
  ptcloud/NN.mat   ตัวแปร xyzPoints (H, W, 3) หน่วย **มิลลิเมตร** ในกรอบกล้อง (z = ระยะ)
  label/NN.png     หน้ากากหลุมที่คนระบาย (ไว้เทียบผล ไม่จำเป็นต่อการตรวจ)

ชุดข้อมูลไม่ได้ให้ intrinsics มาตรง ๆ จึงถอดจากตัวข้อมูลเอง: X/Z เป็นเชิงเส้นกับคอลัมน์ u (และ Y/Z กับแถว v)
→ ความชันคือ 1/fx, 1/fy  แล้ว depth ที่ได้ (z มม.) ก็คือสิ่งที่ analyse_depth ต้องการพอดี

ROS-free: ใช้ได้โดยไม่ต้องมีกล้อง
"""
from typing import Optional, Tuple

import numpy as np


def load_xyz_mat(path: str) -> np.ndarray:
    """อ่าน xyzPoints (H, W, 3) จากไฟล์ .mat (MATLAB v5)"""
    import scipy.io as sio
    m = sio.loadmat(path)
    if 'xyzPoints' not in m:
        raise ValueError(f'{path}: ไม่พบตัวแปร xyzPoints (มี: {[k for k in m if not k.startswith("__")]})')
    P = np.asarray(m['xyzPoints'], dtype=np.float64)
    if P.ndim != 3 or P.shape[2] != 3:
        raise ValueError(f'{path}: xyzPoints ต้องเป็น (H, W, 3) แต่ได้ {P.shape}')
    return P


def depth_and_intrinsics(P: np.ndarray, sample: int = 50) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """(H,W,3) มม. → (depth_mm float32 [0 = ไม่มีข้อมูล], K=(fx, fy, cx, cy))"""
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    valid = np.isfinite(Z) & (Z > 0)
    if valid.sum() < 1000:
        raise ValueError('จุดที่มี depth น้อยเกินไปจะถอด intrinsics ได้')
    H, W = Z.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    a = np.polyfit(u[valid][::sample], (X[valid] / Z[valid])[::sample], 1)
    b = np.polyfit(v[valid][::sample], (Y[valid] / Z[valid])[::sample], 1)
    fx, cx = 1.0 / a[0], -a[1] / a[0]
    fy, cy = 1.0 / b[0], -b[1] / b[0]
    depth = np.where(valid, Z, 0.0).astype(np.float32)
    return depth, (float(fx), float(fy), float(cx), float(cy))


def label_agreement(label: np.ndarray, depth_mm: np.ndarray, K, res, grid=None,
                    pothole_mm: float = 4.0, step: int = 3) -> dict:
    """เทียบหน้ากากหลุมที่คนระบายกับพื้นที่ที่ระบบตีว่าผิดปกติ (ในพิกัดภาพเดียวกัน)

    นับ 2 แบบ: 'depressed' = เซลล์ที่ต่ำกว่าฐานเฉพาะที่ ≥ pothole_mm (ตัวเลขทางกายภาพ)
                'risk>=watch' = เซลล์ที่คะแนนความเสี่ยง ≥ 0.35 (มีขอบเตือนบานออก precision จึงต่ำกว่า)
    ให้ recall / precision / IoU และจำนวนพิกเซลป้ายที่อยู่ในพื้นที่สำรวจ
    """
    from floor_inspect import surface_core as sc
    lab = np.asarray(label) > 0
    H, W = depth_mm.shape
    fx, fy, cx, cy = K
    vv, uu = np.mgrid[0:H:step, 0:W:step]
    z = depth_mm[vv, uu]
    ok = z > 0
    pts = np.c_[((uu - cx) / fx * z)[ok], ((vv - cy) / fy * z)[ok], z[ok]] / 1000.0
    pf = sc.transform_points(pts, res.T_floor_cam)
    r, c = res.dem.rc_of(pf[:, 0], pf[:, 1])
    r = np.floor(r).astype(int)
    c = np.floor(c).astype(int)
    h, w = res.dem.z.shape
    inb = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    L = lab[vv, uu][ok][inb]
    r, c = r[inb], c[inb]
    masks = {'depressed': res.dev_local_mm <= -pothole_mm}
    if grid is not None:
        masks['risk>=watch'] = np.nan_to_num(grid, nan=0.0) >= 0.35
    out = {}
    for name, m in masks.items():
        P = m[r, c]
        tp = float((P & L).sum())
        out[name] = dict(recall=tp / max(L.sum(), 1), precision=tp / max(P.sum(), 1),
                         iou=tp / max((P | L).sum(), 1))
    out['label_px_in_survey'] = int(L.sum())
    out['label_px_total'] = int(lab[::step, ::step].sum())
    return out
