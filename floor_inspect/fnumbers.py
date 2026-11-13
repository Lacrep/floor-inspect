# -*- coding: utf-8 -*-
"""fnumbers — ดัชนีความเรียบพื้นมาตรฐานอุตสาหกรรม จาก DEM (ROS-free)

1) F-Numbers ตาม ASTM E1155/E1155M — ตัวเลขที่ใช้เขียนในสัญญาก่อสร้างพื้นโกดังทั่วโลก
     FF (Flatness)  = ความ "ไม่เป็นคลื่น" ระยะสั้น — วัดจาก "ความโค้ง" q บนฐาน 600 mm
     FL (Levelness) = ความ "ได้ระดับ" ระยะยาว   — วัดจากผลต่างระดับ z บนฐาน 3000 mm
   ทั้งคู่นิยามให้ "ยิ่งมากยิ่งดี":  FF = 114.6 / S_q ,  FL = 317.5 / S_z   (S เป็น mm)
   (ฉบับนิ้ว: FF = 4.57/S_q , FL = 12.5/S_z เมื่อ S เป็นนิ้ว — คูณ 25.4 ได้ค่าเมตริกข้างบน)

2) ช่องใต้ไม้บรรทัด (straightedge gap) แบบ TR34/ACI — ค่าที่ช่างวัดกันหน้างานด้วยไม้บรรทัด
   2 m + ลิ่มวัด ข้อดีคือใช้พื้นที่สั้น ๆ ก็วัดได้ ต่างจาก FL ที่ต้องมีเส้นยาว 3 m ขึ้นไป

⚠ ข้อจำกัดที่ต้องพูดให้ชัดในรายงาน: กล้องตัวเดียวมองพื้นได้กว้าง ~1-2 m ต่อเฟรม
  ซึ่ง "สั้นกว่า" ฐาน 3048 mm ที่ FL ต้องใช้ ฉะนั้นค่า FL จะคำนวณให้เฉพาะเมื่อเส้นวัดยาวพอ
  จริง ๆ เท่านั้น ไม่งั้นคืน None พร้อมเหตุผล — ห้ามรายงานตัวเลขที่ไม่เข้าเกณฑ์มาตรฐาน
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

READING_SPACING_M = 0.300          # ระยะห่างจุดอ่านตามมาตรฐาน
FF_CONST_MM = 114.6                # FF = 114.6 / S_q   (S_q หน่วย mm)
FL_CONST_MM = 317.5                # FL = 317.5 / S_z   (S_z หน่วย mm)
FL_BASE_M = 3.048                  # ฐานวัด levelness = 10 ฟุต
MIN_FF_READINGS = 12               # ต้องมี q อย่างน้อยพอสมควรถึงจะมีนัยสำคัญทางสถิติ
MIN_FL_READINGS = 6


@dataclass
class FNumberResult:
    FF: Optional[float] = None
    FL: Optional[float] = None
    n_q: int = 0
    n_z: int = 0
    Sq_mm: float = float('nan')
    Sz_mm: float = float('nan')
    line_len_m: float = 0.0
    n_lines: int = 0
    note: str = ''

    def to_dict(self):
        return dict(FF=None if self.FF is None else round(self.FF, 1),
                    FL=None if self.FL is None else round(self.FL, 1),
                    n_q=self.n_q, n_z=self.n_z,
                    Sq_mm=round(self.Sq_mm, 3), Sz_mm=round(self.Sz_mm, 3),
                    line_len_m=round(self.line_len_m, 2), n_lines=self.n_lines,
                    note=self.note)


def _reject_3sigma(v: np.ndarray) -> np.ndarray:
    """ตัด outlier เกิน 3 เท่าส่วนเบี่ยงเบนมาตรฐาน (มาตรฐานกำหนดให้ทำหนึ่งรอบ)"""
    if v.size < 3:
        return v
    m, s = float(v.mean()), float(v.std(ddof=1))
    if s <= 0:
        return v
    keep = np.abs(v - m) <= 3.0 * s
    return v[keep] if keep.sum() >= 3 else v


def _profiles_from_dem(dem, axis: int, spacing_m: float, line_gap_m: float = 0.15):
    """ดึง "เส้นวัด" จาก DEM: อ่านความสูงทุก spacing_m ตามแนวที่กำหนด

    axis=0 → เส้นวิ่งตามแนว x (หน้า-หลัง);  axis=1 → ตามแนว y (ซ้าย-ขวา)
    คืน list ของ (ระยะเริ่ม, array ค่าความสูง mm) โดยเส้นที่มีจุดขาดจะถูกตัดเป็นท่อนต่อเนื่อง
    """
    z = dem.z * 1000.0
    step = max(1, int(round(spacing_m / dem.cell)))
    gap = max(1, int(round(line_gap_m / dem.cell)))
    out = []
    if axis == 0:
        for c in range(0, dem.shape[1], gap):
            col = z[::step, c]
            out += _split_runs(col)
    else:
        for r in range(0, dem.shape[0], gap):
            row = z[r, ::step]
            out += _split_runs(row)
    return out


def _split_runs(v: np.ndarray, min_len: int = 4) -> List[np.ndarray]:
    """แยกเป็นช่วงที่ข้อมูลต่อเนื่อง (ไม่มี NaN) — ห้ามข้ามช่องว่างเพราะจะได้ความโค้งเก๊"""
    ok = np.isfinite(v)
    runs, start = [], None
    for i, o in enumerate(ok):
        if o and start is None:
            start = i
        elif not o and start is not None:
            if i - start >= min_len:
                runs.append(v[start:i])
            start = None
    if start is not None and len(v) - start >= min_len:
        runs.append(v[start:])
    return runs


def f_numbers(dem, spacing_m: float = READING_SPACING_M,
              line_gap_m: float = 0.15) -> FNumberResult:
    """คำนวณ FF / FL ตาม ASTM E1155 จาก DEM

    FF: q = ผลต่างของผลต่างระดับที่ห่างกัน 300 mm (= ความโค้งบนฐาน 600 mm)
    FL: z = ผลต่างระดับบนฐาน 3048 mm
    """
    qs, zs = [], []
    max_len = 0.0
    n_lines = 0
    n_per_fl = int(round(FL_BASE_M / spacing_m))       # = 10 ช่วงที่ระยะ 300 mm
    for axis in (0, 1):
        for prof in _profiles_from_dem(dem, axis, spacing_m, line_gap_m):
            n_lines += 1
            max_len = max(max_len, (len(prof) - 1) * spacing_m)
            d = np.diff(prof)                          # ผลต่างระดับทุก 300 mm
            if d.size >= 2:
                qs.append(np.diff(d))                  # ความโค้ง
            if prof.size > n_per_fl:
                zs.append(prof[n_per_fl:] - prof[:-n_per_fl])

    res = FNumberResult(line_len_m=max_len, n_lines=n_lines)
    if qs:
        q = _reject_3sigma(np.concatenate(qs))
        res.n_q = int(q.size)
        if q.size >= MIN_FF_READINGS:
            sq = float(q.std(ddof=1))
            res.Sq_mm = sq
            res.FF = FF_CONST_MM / sq if sq > 1e-9 else float('inf')
    if zs:
        zz = _reject_3sigma(np.concatenate(zs))
        res.n_z = int(zz.size)
        if zz.size >= MIN_FL_READINGS:
            sz = float(zz.std(ddof=1))
            res.Sz_mm = sz
            res.FL = FL_CONST_MM / sz if sz > 1e-9 else float('inf')

    notes = []
    if res.FF is None:
        notes.append(f'FF: ข้อมูลไม่พอ (ต้องมีค่าความโค้ง ≥{MIN_FF_READINGS}, มี {res.n_q})')
    if res.FL is None:
        notes.append(f'FL: ต้องมีเส้นวัดยาว ≥{FL_BASE_M:.3f} m แต่ยาวสุด {max_len:.2f} m '
                     '— กล้องตัวเดียวมองไม่ครอบคลุมพอ ต้องต่อภาพหลายจุด (mosaic) ก่อน')
    res.note = ' · '.join(notes)
    return res


# ───────────────────── ช่องใต้ไม้บรรทัดเสมือน (straightedge) ─────────────────────

def straightedge_gap_mm(dem, length_m: float = 2.0) -> Tuple[np.ndarray, float]:
    """ช่องว่างสูงสุดใต้ไม้บรรทัดยาว length_m ที่วางทาบทุกตำแหน่ง/สองแนว (mm)

    นิยาม: วางไม้บรรทัดพาดจากจุด A ถึง B (ห่างกัน length_m) แล้ววัดระยะจากผิวพื้นถึง
    เส้นตรง AB — ค่าบวก = พื้นต่ำกว่าไม้บรรทัด (มีช่องลอด), ค่าลบ = พื้นนูนดันไม้บรรทัด

    เป็นเกณฑ์ที่ช่างใช้จริงหน้างานและอยู่ในข้อกำหนด TR34/ACI 117 — ข้อดีคือใช้พื้นที่
    สั้นแค่ length_m ก็วัดได้ ต่างจาก FL ที่ต้องการเส้นยาว 3 m ขึ้นไป
    คืน (แผนที่ช่องว่างสูงสุดต่อเซลล์, ค่าสูงสุดทั้งผืน)
    """
    z = dem.z * 1000.0
    n = max(2, int(round(length_m / dem.cell)))
    half = n // 2
    out = np.full(dem.shape, np.nan, np.float32)

    for axis in (0, 1):
        zz = z if axis == 0 else z.T
        H, W = zz.shape
        if H < n + 1:
            continue
        # ปลายไม้บรรทัดสองข้างของแต่ละจุดกึ่งกลาง
        a = zz[:H - 2 * half, :]
        b = zz[2 * half:, :]
        chord = 0.5 * (a + b)                       # ระดับของไม้บรรทัดตรงจุดกึ่งกลาง
        mid = zz[half:H - half, :]
        gap = chord - mid                           # บวก = พื้นต่ำกว่าไม้บรรทัด
        tgt = out if axis == 0 else out.T
        seg = tgt[half:H - half, :]
        with np.errstate(invalid='ignore'):
            tgt[half:H - half, :] = np.where(np.isnan(seg), gap,
                                             np.where(np.abs(gap) > np.abs(seg), gap, seg))
    valid = dem.valid & np.isfinite(out)
    out[~valid] = np.nan
    worst = float(np.nanmax(np.abs(out))) if valid.any() else float('nan')
    return out, worst


# ───────────────────────── เกณฑ์ตัดสิน ─────────────────────────

# ระดับความเรียบมาตรฐานอุตสาหกรรม (ASTM E1155 / ACI 117) — ใช้บอกว่าพื้นที่วัดได้อยู่ชั้นไหน
FF_CLASSES = [
    (20, 'Conventional — พื้นทั่วไป ไม่เหมาะกับหุ่นความเร็วสูง'),
    (25, 'Moderately flat — พื้นโกดังทั่วไป'),
    (35, 'Flat — เหมาะกับ AMR ส่วนใหญ่'),
    (45, 'Very flat — รถยกทางแคบ (VNA)'),
    (60, 'Super flat — ระบบอัตโนมัติความแม่นยำสูง'),
]


def classify_FF(ff: Optional[float]) -> str:
    if ff is None or not np.isfinite(ff):
        return 'ไม่มีข้อมูลพอ'
    label = FF_CLASSES[0][1]
    for thr, name in FF_CLASSES:
        if ff >= thr:
            label = name
    if ff < FF_CLASSES[0][0]:
        return f'ต่ำกว่า Conventional (FF<{FF_CLASSES[0][0]}) — พื้นไม่เรียบอย่างมีนัยสำคัญ'
    return label
