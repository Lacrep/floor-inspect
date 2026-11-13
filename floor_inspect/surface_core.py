# -*- coding: utf-8 -*-
"""surface_core — แกนหลักของการวัด "ความไม่เรียบของพื้น" จาก depth ของ D455 (ROS-free)

ท่อ (pipeline) ตามงาน floor-flatness scanning จริง (ASTM E1155 / iscano deviation map)
และงาน robot-assisted floor profiling (Remote Sens. 2019, 11(22), 2626):

    depth (mm) ──deproject──▶ point cloud (camera optical frame)
               ──robust plane fit──▶ ระนาบอ้างอิง (reference plane)
               ──transform──▶ floor frame (x=หน้า, y=ซ้าย, z=ขึ้น, z=0 ที่ระนาบ)
               ──rasterise──▶ DEM: กริดมองจากบน เก็บ "median z ต่อเซลล์"
               ──top-hat──▶ dev_local (หลุม/เนินเฉพาะที่)  +  dev_plane (ระดับรวม/ลาดเอียง)
               ──cluster──▶ รายการ defect พร้อมขนาด/ความลึก

หัวใจของความแม่นยำอยู่ที่ 2 จุด:
  1. plane fit ต้อง "ไม่โดนหลุมดึง" → RANSAC หาระนาบตั้งต้น แล้ว refine ด้วย trimmed
     least-squares (IRLS ตัด outlier ด้วย MAD) ซ้ำหลายรอบ ระนาบจึงนิ่งระดับ sub-mm
  2. per-cell MEDIAN ไม่ใช่ mean → noise ต่อจุดของ D455 (~2-3mm ที่ 1m) ถูกกดลงเหลือ
     <1mm ต่อเซลล์เมื่อมีจุดหลายสิบจุดต่อเซลล์
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np

try:                                   # open3d ใช้แค่ RANSAC ตั้งต้น (เร็วเพราะเป็น C++)
    import open3d as o3d
    _HAS_O3D = True
except Exception:                      # ไม่มีก็ยังทำงานได้ ด้วย RANSAC เวอร์ชัน numpy ข้างล่าง
    _HAS_O3D = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ───────────────────────────── โครงสร้างข้อมูล ─────────────────────────────

@dataclass
class Plane:
    """ระนาบ n·p + d = 0 โดย n เป็นเวกเตอร์หนึ่งหน่วยที่ชี้ "ขึ้น" (ออกจากพื้นไปทางกล้อง)"""
    n: np.ndarray
    d: float
    n_inliers: int = 0
    n_total: int = 0
    rms_mm: float = float('nan')       # RMS ของ inlier รอบระนาบ = ความ "แบน" ของ fit
    ok: bool = False

    @property
    def inlier_frac(self) -> float:
        return self.n_inliers / self.n_total if self.n_total else 0.0

    @property
    def cam_height_m(self) -> float:
        """ระยะตั้งฉากจากกล้อง (origin) ถึงระนาบ"""
        return abs(self.d)

    def signed_dist(self, xyz: np.ndarray) -> np.ndarray:
        return xyz @ self.n + self.d


@dataclass
class DEM:
    """Digital Elevation Model มองจากด้านบนใน floor frame

    การวางแกนให้ภาพ "อ่านแล้วเข้าใจทันที": แถวบนสุด = ไกลจากกล้องที่สุด (x มาก),
    คอลัมน์ซ้ายสุด = ด้านซ้ายของหุ่น (y มาก) → ภาพ top-down หันหน้าขึ้นเหมือนแผนที่
    """
    z: np.ndarray                      # (H,W) float32 เมตร, NaN = ไม่มีข้อมูล
    count: np.ndarray                  # (H,W) int32 จำนวนจุดต่อเซลล์
    spread_mm: np.ndarray              # (H,W) float32 IQR-based spread ต่อเซลล์ (คุณภาพข้อมูล)
    cell: float                        # ขนาดเซลล์ (m)
    x_max: float                       # ขอบไกลสุด (m) → แถว 0
    y_max: float                       # ขอบซ้ายสุด (m) → คอลัมน์ 0

    @property
    def shape(self):
        return self.z.shape

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.z)

    @property
    def coverage(self) -> float:
        return float(self.valid.mean()) if self.z.size else 0.0

    @property
    def area_m2(self) -> float:
        return float(self.valid.sum()) * self.cell * self.cell

    def xy_of(self, r, c):
        """(row, col) → (x, y) เมตร ใน floor frame (จุดกึ่งกลางเซลล์)"""
        x = self.x_max - (np.asarray(r) + 0.5) * self.cell
        y = self.y_max - (np.asarray(c) + 0.5) * self.cell
        return x, y

    def rc_of(self, x, y):
        r = ((self.x_max - np.asarray(x)) / self.cell).astype(np.int32)
        c = ((self.y_max - np.asarray(y)) / self.cell).astype(np.int32)
        return r, c

    def extent_m(self) -> Tuple[float, float]:
        """(ยาวตามแนวหน้า, กว้างตามแนวข้าง) ของกริดทั้งผืน"""
        H, W = self.z.shape
        return H * self.cell, W * self.cell


@dataclass
class Defect:
    kind: str                          # 'pothole' | 'bump' | 'step' | 'missing' | 'obstacle'
    x: float                           # จุดศูนย์กลางใน floor frame (m) — x=หน้า
    y: float                           # y=ซ้าย (m)
    length_m: float                    # ขนาดตามแนวหน้า
    width_m: float                     # ขนาดตามแนวข้าง
    area_m2: float
    peak_mm: float                     # ค่าเบี่ยงเบนสูงสุด (ลบ=หลุม, บวก=นูน)
    mean_mm: float
    cells: int
    bbox: Tuple[int, int, int, int]    # (r0, c0, r1, c1) ใน DEM
    edge_step_mm: float = 0.0          # ต่างระดับสูงสุดที่ "ขอบ" ของ defect นี้ (ขอบคม = อันตรายกว่า)
    noise_mm: float = 0.0              # noise เฉลี่ยบริเวณนี้ — ใช้บอกว่าเชื่อค่าได้แค่ไหน
    snr: float = 0.0                   # |peak| / noise → ต่ำกว่า ~3 คือเริ่มไม่น่าเชื่อ
    range_m: float = 0.0               # ระยะจากกล้องถึง defect (โดยประมาณ)
    risk: str = 'ok'                   # เติมโดย risk.py
    risk_score: float = 0.0
    safe_speed_ms: float = 0.0         # ความเร็วสูงสุดที่ข้ามจุดนี้แล้วสั่นไม่เกินลิมิต
    reason: str = ''

    @property
    def min_extent_m(self) -> float:
        """ด้านที่แคบที่สุด = "ความกว้างช่องที่ล้อต้องข้าม" (ใช้คิดฟิสิกส์ล้อตกหลุม)"""
        return min(self.length_m, self.width_m)

    def to_dict(self):
        return dict(kind=self.kind, x=round(self.x, 4), y=round(self.y, 4),
                    length_m=round(self.length_m, 4), width_m=round(self.width_m, 4),
                    area_m2=round(self.area_m2, 5), peak_mm=round(self.peak_mm, 2),
                    mean_mm=round(self.mean_mm, 2), cells=self.cells,
                    edge_step_mm=round(self.edge_step_mm, 2), noise_mm=round(self.noise_mm, 2),
                    snr=round(self.snr, 2), range_m=round(self.range_m, 3),
                    risk=self.risk, risk_score=round(self.risk_score, 3),
                    safe_speed_ms=round(self.safe_speed_ms, 3), reason=self.reason)


@dataclass
class SurfaceParams:
    """พารามิเตอร์ทั้งท่อ — ค่า default ตั้งไว้สำหรับ D455 มองพื้นจากสูง ~0.4-1.2 m"""
    # --- deprojection ---
    zmin: float = 0.25                 # D455 วัดใกล้สุดได้จริง ~0.28 m (วัดเองแล้ว)
    zmax: float = 3.0
    stride_fit: int = 4                # stride สำหรับ plane fit (ไม่ต้องใช้ทุกพิกเซล)
    pixel_stride: int = 1              # ย่อภาพ depth ก่อนวิเคราะห์ (1 = ใช้ทุกพิกเซล)
                                       # 2 = เร็วขึ้น ~3 เท่า เสียความไวเล็กน้อย
                                       # ใช้ตอน align depth ขึ้นกริดสีที่ใหญ่กว่าจริง

    # --- plane fit ---
    ransac_dist: float = 0.02          # m — เผื่อ noise ตอนหาระนาบตั้งต้น
    ransac_iters: int = 300
    refine_iters: int = 4              # รอบ trimmed refit
    refine_k_mad: float = 2.5          # ตัด outlier เกิน k*MAD ออกจากการ fit
    min_inlier_frac: float = 0.25
    max_plane_rms_mm: float = 12.0     # fit ที่กระจายกว่านี้ = ไม่ใช่ "พื้นเรียบ" จริง
    max_usable_mm: float = 15.0        # เกินนี้ = "วัดไม่ได้แล้ว" ไม่ใช่ "พื้นดี"
                                       # ตัดบริเวณนั้นออกจากพื้นที่สำรวจ (ยังรายงาน
                                       # สิ่งกีดขวางใหญ่อยู่ แต่เลิกอ้างว่าวัดหลุมได้)
    rough_win_m: float = 0.15          # หน้าต่างวัดความขรุขระประจำถิ่น
    rough_open_m: float = 0.40         # หน้าต่าง opening ที่ใช้ลบ "ยอดที่ defect สร้างเอง"
                                       # ออกจากแผนที่ความขรุขระ (0 = ใช้เส้น fit อย่างเดียว)
    texture_prior: Optional[Tuple[float, float]] = None   # (a, b) จากเฟรมก่อน
    texture_alpha: float = 0.30        # น้ำหนักของเฟรมใหม่ใน EMA
    k_rough: float = 3.0               # defect ต้องโผล่เหนือเนื้อผิวกี่เท่าของ MAD
                                       # 3.0 คือหัวเข่าที่วัดได้: พรมเปล่า 25->4 จุด
                                       # ส่วนพื้นเรียบกับหลุมจริงไม่กระทบเลย
    min_points_cell: float = 6.0       # จุด depth เฉลี่ยต่อเซลล์ (ในหน้าต่าง solid_win_m)
                                       # ต่ำกว่านี้ = ตัดออกจากพื้นที่สำรวจ
                                       # median ของเซลล์ที่มี 2-4 จุด กรอง outlier ไม่ได้เลย
                                       # ทั้งก้อนที่มุมเฉียงจะพลาดพร้อมกัน (stereo error
                                       # สัมพันธ์กันเชิงพื้นที่) กลายเป็นหลุม/เนินผีทั้งหย่อม
                                       # วัดจริงบนพรม 12 เฟรม: เซลล์ของหลุมผี -37 mm มี 4.1 จุด
                                       # ส่วนเนินจริง +24 mm มี 14 จุด (ประชากร p10/p50 = 4.4/9)
                                       # ตัดที่ 6 จุด: จุดปลอมไกล 1.75 -> 0.00 ต่อเฟรม
                                       # sd ของจำนวนที่รายงาน 2.18 -> 1.25 เนินจริงยังเจอ 12/12
                                       # เสียพื้นที่ 1.72 -> 1.31 m² แต่ส่วนที่เสียคือส่วนที่
                                       # พ่นจุดปลอมล้วน ๆ (ลองตัดด้วยมุมตกกระทบ 40° แทน
                                       # เหลือพื้นที่แค่ 0.98 m² และยังมีจุดปลอมเหลือ)
                                       # เกณฑ์นี้ปรับตัวเองตามกล้อง/ความสูง/ระยะ/cell/stride
                                       # เพราะความหนาแน่นจริงรวมทุกอย่างไว้แล้ว (0 = ปิด)
    min_survey_m2: float = 0.15        # สำรวจได้น้อยกว่านี้ = ยังไม่ใช่การตรวจพื้น
                                       # (แผ่นเรียบเล็ก ๆ ใกล้เลนส์ก็ผ่านเกณฑ์ rms ได้)

    # --- DEM ---
    cell: float = 0.01                 # 10 mm/เซลล์ — ที่ 1 m กล้องให้ ~2-3 mm/พิกเซล
    dem_max_x: float = 3.0             # ตัดกริดไม่ให้บานตอนเห็นไกล
    dem_max_y: float = 2.0
    min_cell_points: int = 2           # เซลล์ที่มีจุดน้อยกว่านี้ถือว่าไม่น่าเชื่อ
    solid_frac: float = 0.7            # ความหนาแน่นข้อมูลรอบเซลล์ขั้นต่ำที่ถือว่า "สำรวจได้จริง"
    solid_win_m: float = 0.10          # หน้าต่างวัดความหนาแน่นนั้น

    # --- แยกสเกล (local defect vs global levelness) ---
    base_win_m: float = 0.40           # ยาว "ไม้บรรทัดเสมือน" ที่ใช้หาฐานอ้างอิงเฉพาะที่
                                       # ความผิดปกติที่กว้างกว่านี้จะไปโผล่ในแผนที่ระดับรวม
                                       # (levelness) แทน — ตรงตามนิยาม FF vs FL ของ ASTM
    median_pre: int = 3                # median filter บน DEM ก่อนวิเคราะห์ (0=ปิด) — กัน noise เซลล์เดี่ยว
    noise_win_m: float = 0.15          # หน้าต่างประมาณ noise เฉพาะที่

    # --- เกณฑ์ตรวจจับ defect ---
    # threshold จริงที่ใช้ = max(ค่าคงที่, k_sigma × noise ที่วัดได้ตรงนั้น)
    # เหตุผล: ที่ 0.5 m กล้องนิ่งพอจะเห็นหลุม 2 mm แต่ที่ 1.8 m noise ต่อจุดเกิน 6 mm
    # ถ้าใช้ threshold เดียวทั้งภาพ จะได้ false positive เต็มไปหมดที่ระยะไกล (พิสูจน์แล้วด้วย synth)
    k_sigma: float = 4.0
    pothole_mm: float = 4.0            # ลึกกว่านี้ = หลุม
    bump_mm: float = 4.0               # นูนกว่านี้ = เนิน
    step_mm: float = 3.0               # ต่างระดับบนฐานวัด step_base_m = ลิ่ม/รอยต่อ (iscano: 3mm ทำล้อ caster พัง)
    step_base_m: float = 0.04          # ฐานวัด step — สั้นไปเจอแต่ noise, ยาวไปขอบคมจะเบลอ
    min_defect_area_m2: float = 0.0009 # 900 mm² (~3x3 เซลล์) กันจุด noise เดี่ยว ๆ
    min_snr: float = 3.0               # |peak|/noise ต่ำกว่านี้ = ไม่ยืนยัน
    wheel_radius_m: float = 0.0375     # รัศมีล้อ caster ที่ใช้ตัดสิน (Ø75 mm)
    min_effective_mm: float = 2.0      # ล้อต้องจม/ยกจริงเกินนี้ถึงจะนับเป็น defect
                                       # (0 = ปิด รายงานทุกอย่างที่เกินเกณฑ์ความลึก)
    obstacle_mm: float = 60.0          # สูงกว่านี้ไม่ใช่ "พื้นไม่เรียบ" แต่เป็นสิ่งกีดขวาง
    missing_min_area_m2: float = 0.002 # รูข้อมูลใหญ่กว่านี้ = น่าสงสัย (หลุมลึกทำ depth หลุด)


# ───────────────────────────── deprojection ─────────────────────────────

def deproject_full(depth_mm: np.ndarray, K, zmin: float, zmax: float):
    """คืน xyz รูปทรง (H,W,3) เมตร + mask ความถูกต้อง — คงรูปภาพไว้เพื่อวาด overlay แบบทึบทั้งภาพ

    K = (fx, fy, cx, cy) ของ "ภาพสี" (เพราะใช้ aligned_depth_to_color)
    """
    fx, fy, cx, cy = K
    H, W = depth_mm.shape
    z = depth_mm.astype(np.float32) * 0.001
    valid = (z > zmin) & (z < zmax)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return np.dstack([x, y, z]).astype(np.float32), valid


# ───────────────────────────── plane fitting ─────────────────────────────

def _plane_from_points(pts: np.ndarray):
    """PCA fit: คืน (n, d) โดย n = eigenvector ของแกนที่แปรปรวนน้อยสุด"""
    c = pts.mean(axis=0)
    q = pts - c
    # SVD เสถียรกว่าการทำ eigh บน covariance เมื่อจุดเกือบ coplanar
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    n = vt[2]
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return None
    n = n / nn
    d = -float(n @ c)
    return n, d


def _ransac_numpy(pts: np.ndarray, dist: float, iters: int, rng) -> Optional[Tuple[np.ndarray, float]]:
    """RANSAC สำรองเมื่อไม่มี open3d — vectorised พอที่จะรันได้จริง"""
    n_pts = len(pts)
    if n_pts < 3:
        return None
    best_cnt, best = -1, None
    idx = rng.integers(0, n_pts, size=(iters, 3))
    for tri in idx:
        p0, p1, p2 = pts[tri]
        nv = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(nv)
        if nn < 1e-9:
            continue
        nv = nv / nn
        dv = -float(nv @ p0)
        cnt = int(np.count_nonzero(np.abs(pts @ nv + dv) < dist))
        if cnt > best_cnt:
            best_cnt, best = cnt, (nv, dv)
    return best


def fit_reference_plane(xyz: np.ndarray, p: SurfaceParams, seed: int = 0) -> Plane:
    """หาระนาบพื้นอ้างอิงแบบทนหลุม/เนิน

    ทำไมต้อง refine หลัง RANSAC: RANSAC ให้ระนาบที่ "จุดส่วนใหญ่อยู่ใกล้" ก็จริง แต่ threshold
    20 mm ของมันทำให้หลุมลึก 5 mm ยังนับเป็น inlier และดึงระนาบเอียงได้ 1-2 mrad ซึ่งพอจะทำให้
    ค่าเบี่ยงเบนที่ปลายกริดผิดไปหลายมิลลิเมตร — trimmed refit ตัดพวกนี้ทิ้งทีละรอบ
    """
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n_total = len(pts)
    if n_total < 200:
        return Plane(n=np.array([0.0, -1.0, 0.0]), d=0.0, n_total=n_total, ok=False)

    seed_plane = None
    if _HAS_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        try:
            model, inl = pcd.segment_plane(distance_threshold=p.ransac_dist,
                                           ransac_n=3, num_iterations=p.ransac_iters)
            a, b, c, d = model
            nv = np.array([a, b, c], dtype=np.float64)
            nn = np.linalg.norm(nv)
            if nn > 1e-9 and len(inl) >= 3:
                seed_plane = (nv / nn, float(d) / nn)
        except Exception:
            seed_plane = None
    if seed_plane is None:
        seed_plane = _ransac_numpy(pts, p.ransac_dist, p.ransac_iters,
                                   np.random.default_rng(seed))
    if seed_plane is None:
        return Plane(n=np.array([0.0, -1.0, 0.0]), d=0.0, n_total=n_total, ok=False)

    n, d = seed_plane
    # กล้องมองพื้น → พื้นอยู่ "ใต้" กล้องเสมอ ในเฟรม optical แกน y ชี้ลง
    # ดังนั้นเวกเตอร์ "ขึ้น" ต้องมี y ติดลบ — กฎนี้เชื่อถือได้เพราะโจทย์คือกล้องมองพื้นแน่นอน
    if n[1] > 0:
        n, d = -n, -d

    resid = pts @ n + d
    keep = np.abs(resid) < p.ransac_dist
    for _ in range(max(0, p.refine_iters)):
        sel = pts[keep]
        if len(sel) < 100:
            break
        fit = _plane_from_points(sel)
        if fit is None:
            break
        n_new, d_new = fit
        if n_new[1] > 0:
            n_new, d_new = -n_new, -d_new
        resid = pts @ n_new + d_new
        r_in = resid[keep]
        mad = float(np.median(np.abs(r_in - np.median(r_in)))) * 1.4826
        thr = max(p.refine_k_mad * mad, 0.002)      # อย่างน้อย 2 mm กันหดจนเหลือจุดเดียว
        keep_new = np.abs(resid) < thr
        n, d = n_new, d_new
        if keep_new.sum() < 100:
            break
        if keep_new.sum() == keep.sum() and np.array_equal(keep_new, keep):
            keep = keep_new
            break
        keep = keep_new

    n_in = int(keep.sum())
    rms_mm = float(np.sqrt(np.mean((resid[keep] * 1000.0) ** 2))) if n_in else float('nan')
    return Plane(n=n, d=float(d), n_inliers=n_in, n_total=n_total,
                 rms_mm=rms_mm, ok=(n_in / n_total) >= p.min_inlier_frac)


def floor_transform(plane: Plane) -> np.ndarray:
    """สร้าง T (4x4) แปลง camera optical frame → floor frame

    floor frame: x = ทิศที่กล้องมอง (ฉายลงระนาบ), y = ซ้าย, z = ขึ้น, ระนาบพื้นอยู่ที่ z = 0
    ⇒ หลังแปลงแล้ว ค่า z ของแต่ละจุด "คือ" ค่าเบี่ยงเบนจากระนาบอ้างอิงโดยตรง
    """
    z_axis = plane.n / np.linalg.norm(plane.n)
    fwd = np.array([0.0, 0.0, 1.0])                       # แกนมองของกล้อง
    x_axis = fwd - np.dot(fwd, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:                     # กล้องมองตั้งฉากพื้นเป๊ะ
        x_axis = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R_cf = np.stack([x_axis, y_axis, z_axis], axis=1)     # คอลัมน์ = แกน floor ในพิกัดกล้อง
    origin = -plane.d * plane.n                           # เท้าตั้งฉากของกล้องบนระนาบ
    T = np.eye(4)
    T[:3, :3] = R_cf.T
    T[:3, 3] = -R_cf.T @ origin
    return T


def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """รองรับทั้ง (N,3) และ (H,W,3)"""
    a = np.asarray(pts)
    flat = a.reshape(-1, 3)
    out = flat @ T[:3, :3].T + T[:3, 3]
    return out.reshape(a.shape)


# ───────────────────────────── DEM rasterisation ─────────────────────────────

def build_dem(pts_floor: np.ndarray, p: SurfaceParams,
              bounds: Optional[Tuple[float, float, float, float]] = None) -> Optional[DEM]:
    """จุดใน floor frame → กริดมองจากบน เก็บ median z ต่อเซลล์

    ใช้ median (ไม่ใช่ mean) เพราะ depth ของกล้อง stereo มี outlier แบบ "ยิงหลุด" ปนมาเสมอ
    ค่า median ต่อเซลล์ยังลด noise แบบสุ่มลงตามจำนวนจุดในเซลล์ด้วย
    """
    pts = np.asarray(pts_floor, dtype=np.float32).reshape(-1, 3)
    if len(pts) < 50:
        return None
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    if bounds is None:
        # ตัดขอบด้วย percentile กัน outlier ตัวเดียวทำกริดบานเป็นสิบเมตร
        x_lo, x_hi = np.percentile(x, [0.5, 99.5])
        y_lo, y_hi = np.percentile(y, [0.5, 99.5])
        x_hi = min(float(x_hi), x_lo + p.dem_max_x)
        y_lo = max(float(y_lo), -p.dem_max_y)
        y_hi = min(float(y_hi), p.dem_max_y)
    else:
        x_lo, x_hi, y_lo, y_hi = bounds
    if not (np.isfinite([x_lo, x_hi, y_lo, y_hi]).all()) or x_hi <= x_lo or y_hi <= y_lo:
        return None

    # ล็อกขอบกริดให้ตกบนพิกัดโลกที่หารด้วย cell ลงตัว
    # ถ้าไม่ทำ ขอบจะขยับตามพิกเซลไกลสุดที่บังเอิญคืน depth มาในเฟรมนั้น (วัดได้ ±17 mm)
    # ทำให้ทุกเซลล์เลื่อนเฟสไม่เท่ากันทุกเฟรม — ผลวิเคราะห์กะพริบทั้งที่พื้นไม่ขยับ
    x_hi = float(np.ceil(x_hi / p.cell) * p.cell)
    y_hi = float(np.ceil(y_hi / p.cell) * p.cell)
    x_lo = float(np.floor(x_lo / p.cell) * p.cell)
    y_lo = float(np.floor(y_lo / p.cell) * p.cell)

    H = int(np.ceil((x_hi - x_lo) / p.cell))
    W = int(np.ceil((y_hi - y_lo) / p.cell))
    if H < 3 or W < 3 or H * W > 4_000_000:
        return None

    r = ((x_hi - x) / p.cell).astype(np.int32)
    c = ((y_hi - y) / p.cell).astype(np.int32)
    inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    r, c, zz = r[inside], c[inside], z[inside]
    if len(zz) < 50:
        return None

    idx = r.astype(np.int64) * W + c
    ncell = H * W
    counts = np.bincount(idx, minlength=ncell)

    # median + IQR ต่อเซลล์แบบ vectorised: เรียงตาม (เซลล์, z) แล้วหยิบตำแหน่งกลางของแต่ละกลุ่ม
    order = np.lexsort((zz, idx))
    idx_s, z_s = idx[order], zz[order]
    starts = np.zeros(ncell, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    nz = counts > 0
    med = np.full(ncell, np.nan, dtype=np.float32)
    spread = np.zeros(ncell, dtype=np.float32)
    if nz.any():
        cs, st = counts[nz], starts[nz]
        med[nz] = z_s[st + cs // 2]
        q1 = z_s[st + cs // 4]
        q3 = z_s[st + (3 * cs) // 4]
        spread[nz] = (q3 - q1) * 1000.0        # mm — IQR ต่อเซลล์ = ตัวชี้คุณภาพข้อมูล

    med[counts < p.min_cell_points] = np.nan   # เซลล์จุดน้อยไม่น่าเชื่อ ตัดทิ้ง
    return DEM(z=med.reshape(H, W), count=counts.reshape(H, W).astype(np.int32),
               spread_mm=spread.reshape(H, W), cell=p.cell,
               x_max=float(x_hi), y_max=float(y_hi))


def solid_mask(dem: DEM, win_m: float = 0.10, min_frac: float = 0.7) -> np.ndarray:
    """เซลล์ที่อยู่ใน "บริเวณข้อมูลแน่น" — ความหนาแน่นรอบตัวถึงเกณฑ์

    ต้องวัดความหนาแน่น *เฉพาะที่* ไม่ใช่สัดส่วนต่อความกว้างกริดทั้งแถว เพราะ FOV กล้อง
    เป็นสี่เหลี่ยมคางหมู — แถวใกล้กล้องเห็นพื้นแคบ ๆ แต่ข้อมูลตรงนั้น "แน่นและดีที่สุด"
    ถ้าวัดแบบต่อแถว จะตัด near field ทิ้งทั้งที่เป็นข้อมูลคุณภาพสูงสุด (เคยพลาดมาแล้ว)
    """
    v = dem.valid.astype(np.float32)
    if not _HAS_CV2:
        return dem.valid
    k = max(3, int(round(win_m / dem.cell)) | 1)
    k = min(k, (min(dem.shape) - 1) | 1, 31)
    dens = cv2.blur(v, (k, k), borderType=cv2.BORDER_REPLICATE)
    return dem.valid & (dens >= min_frac)


def crop_to_surveyed(dem: DEM, min_frac: float = 0.7, win_m: float = 0.10) -> DEM:
    """ตัดกริดให้เหลือกรอบของ "พื้นที่ที่สำรวจได้จริง"

    จำเป็นเพราะที่ระยะไกล จุดโปร่งจนเซลล์ว่างเป็นปกติ ถ้าไม่ตัดทิ้ง ตัวตรวจ "รูข้อมูล"
    จะรายงานความผิดปกติเก๊เป็นสิบจุดตรงขอบไกล (เจอจริงตอนทดสอบกับ synth: 14 จุดปลอม)
    """
    sm = solid_mask(dem, win_m, min_frac)
    if not sm.any():
        return dem
    rr, cc = np.nonzero(sm)
    r0, r1 = int(rr.min()), int(rr.max()) + 1
    c0, c1 = int(cc.min()), int(cc.max()) + 1
    if (r1 - r0) < 3 or (c1 - c0) < 3:
        return dem
    return DEM(z=dem.z[r0:r1, c0:c1].copy(), count=dem.count[r0:r1, c0:c1].copy(),
               spread_mm=dem.spread_mm[r0:r1, c0:c1].copy(), cell=dem.cell,
               x_max=dem.x_max - r0 * dem.cell, y_max=dem.y_max - c0 * dem.cell)


def accumulate_dem(acc: Optional[dict], dem: DEM) -> dict:
    """สะสม DEM หลายเฟรม (ตอนหุ่น/กล้องอยู่นิ่ง) เพื่อกด noise ลงอีก

    เก็บผลรวมและผลรวมกำลังสองต่อเซลล์ → ได้ทั้งค่าเฉลี่ยและ std ต่อเซลล์
    (std ข้ามเฟรมคือ "ความนิ่ง" ที่เอาไปรายงานความน่าเชื่อถือได้ตรง ๆ)
    """
    v = dem.valid
    if acc is None or acc['shape'] != dem.shape or acc['cell'] != dem.cell:
        acc = dict(shape=dem.shape, cell=dem.cell, x_max=dem.x_max, y_max=dem.y_max,
                   s=np.zeros(dem.shape, np.float64), s2=np.zeros(dem.shape, np.float64),
                   n=np.zeros(dem.shape, np.int32), frames=0)
    zz = np.where(v, dem.z, 0.0).astype(np.float64)
    acc['s'] += zz
    acc['s2'] += zz * zz
    acc['n'] += v.astype(np.int32)
    acc['frames'] += 1
    return acc


def finish_accumulation(acc: dict, min_frames: int = 2) -> Tuple[DEM, np.ndarray]:
    """คืน (DEM เฉลี่ย, std ข้ามเฟรมเป็น mm)"""
    n = acc['n']
    ok = n >= min_frames
    mean = np.full(acc['shape'], np.nan, np.float32)
    std_mm = np.full(acc['shape'], np.nan, np.float32)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = acc['s'] / np.maximum(n, 1)
        var = acc['s2'] / np.maximum(n, 1) - m * m
    mean[ok] = m[ok].astype(np.float32)
    std_mm[ok] = (np.sqrt(np.maximum(var[ok], 0.0)) * 1000.0).astype(np.float32)
    dem = DEM(z=mean, count=n.copy(), spread_mm=np.nan_to_num(std_mm), cell=acc['cell'],
              x_max=acc['x_max'], y_max=acc['y_max'])
    return dem, std_mm


# ───────────────────────── แยกสเกล: local defect vs levelness ─────────────────────────

def _fill_nan_nearest(z: np.ndarray, valid: np.ndarray, max_iter: int = 8) -> np.ndarray:
    """เติมช่องว่างด้วยค่าเพื่อนบ้าน (เฉพาะตอนทำ morphology) — ไม่ใช่การ "แต่งข้อมูล"
    เพราะผลลัพธ์ถูก mask กลับด้วย valid เดิมเสมอ"""
    out = z.copy()
    out[~valid] = 0.0
    m = valid.astype(np.float32)
    if not _HAS_CV2:
        return out
    k = np.ones((3, 3), np.float32)
    for _ in range(max_iter):
        if m.min() > 0.5:
            break
        s = cv2.filter2D(out * m, -1, k, borderType=cv2.BORDER_REPLICATE)
        w = cv2.filter2D(m, -1, k, borderType=cv2.BORDER_REPLICATE)
        grow = (w > 0) & (m < 0.5)
        out[grow] = s[grow] / w[grow]
        m[grow] = 1.0
    return out


def median_prefilter(dem: DEM, k: int = 3) -> DEM:
    """median filter บน DEM — ลบ noise เซลล์เดี่ยวโดย "ไม่ทำให้ขอบคมเบลอ"
    (คุณสมบัติเฉพาะของ median: ลิ่ม 3 mm ยังคมอยู่ ต่างจาก mean/gaussian ที่จะไถขอบทิ้ง)"""
    if k < 3 or not _HAS_CV2:
        return dem
    k = int(k) | 1
    v = dem.valid
    zf = _fill_nan_nearest(dem.z, v).astype(np.float32)
    sm = cv2.medianBlur(zf, min(k, 5))          # CV_32F รองรับ ksize 3 และ 5 เท่านั้น
    out = np.where(v, sm, np.nan).astype(np.float32)
    return DEM(z=out, count=dem.count, spread_mm=dem.spread_mm, cell=dem.cell,
               x_max=dem.x_max, y_max=dem.y_max)


# ค่าคงที่แปลง "เศษเหลือจาก median 3×3" → σ ของ noise จริงต่อเซลล์
# คาลิเบรตกับ synth ที่รู้ σ เป๊ะ ๆ: อัตราส่วนคงที่ 0.82 ทุกความสูง/มุมก้ม (0.4-0.8 m, 45-70°)
# → 1.63 × 0.82 ≈ 1.34  (ดู test_surface_core.py::test_noise_estimator ที่ยึดค่านี้ไว้)
_MED3_RESID_TO_SIGMA = 1.34


def local_noise_mm(dem: DEM, win_m: float = 0.15) -> np.ndarray:
    """ประมาณ noise (1σ, mm) ของค่าความสูงต่อเซลล์ แบบวัดจากข้อมูลเองล้วน ๆ

    วิธี: เศษที่เหลือหลัง median 3×3 คือส่วนความถี่สูง = noise เกือบล้วน (ผิวจริงเปลี่ยนช้ากว่า)
    แล้วเอา MAD ของเศษนั้นในหน้าต่างท้องถิ่น → ทนต่อ defect จริงที่โผล่มาปนในหน้าต่าง

    ทำไมไม่ใช้สูตร σ = z²·δ/(f·B) ตรง ๆ: สูตรนั้นไม่รู้จักผิวสะท้อน มุมเฉียง หรือแสง
    ซึ่งเปลี่ยน noise จริงได้เป็นเท่าตัว — วัดจากข้อมูลเองแม่นกว่าและปรับตัวเองได้
    """
    v = dem.valid
    out = np.full(dem.shape, np.nan, np.float32)
    if not v.any() or not _HAS_CV2:
        return np.nan_to_num(out, nan=1.0)
    zf = _fill_nan_nearest(dem.z * 1000.0, v).astype(np.float32)
    resid = np.abs(zf - cv2.medianBlur(zf, 3))
    k = max(3, int(round(win_m / dem.cell)) | 1)
    k = min(k, (min(dem.shape) - 1) | 1, 51)
    # ค่ากลางของ |resid| ในหน้าต่าง — ใช้ blur เป็นตัวประมาณ mean แล้วคูณตัวแปลง
    # (mean ของ |resid| = 0.7979σ_resid สำหรับ noise เกาส์เซียน)
    mean_abs = cv2.blur(resid, (k, k), borderType=cv2.BORDER_REPLICATE)
    sigma = np.maximum(mean_abs / 0.7979 * _MED3_RESID_TO_SIGMA, 0.05)
    out = np.where(v, sigma, np.nan).astype(np.float32)
    return out


def deviation_maps(dem: DEM, p: SurfaceParams, noise_mm: Optional[np.ndarray] = None):
    """คืน (dev_plane_mm, dev_local_mm, base_mm)

    dev_plane  = z เทียบระนาบอ้างอิงตรง ๆ  → "levelness" แผนที่ระดับรวม (สูง/ต่ำทั้งผืน)
                 นี่คือแผนที่แบบเดียวกับ deviation heat map ของงานสแกนพื้นโกดัง
    dev_local  = z − ฐานอ้างอิงเฉพาะที่ → "flatness" ตัดความลาดเอียง/คลื่นยาวออกหมด
    base_mm    = ฐานอ้างอิงเฉพาะที่ ("ไม้บรรทัดเสมือน" ยาว base_win_m)

    ฐานอ้างอิงหาแบบสองรอบ:
      รอบ 1  top-hat เชิงสัณฐาน → ได้ "ผู้ต้องสงสัย" คร่าว ๆ
      รอบ 2  หาค่าเฉลี่ยเฉพาะที่ "โดยไม่นับเซลล์ผู้ต้องสงสัย" → ฐานที่มาจากพื้นดี ๆ ล้วน

    ทำไมต้องรอบ 2: ถ้าใช้ top-hat ล้วน จะพังสองแบบที่เจอจริงกับ synth
      • หลุมสองใบห่างกัน 0.28 m แต่ SE กว้าง 0.40 m → erosion ลากค่าก้นหลุมมาเป็นฐานของ
        พื้นดี ๆ ระหว่างหลุม แล้วรายงาน "เนิน 10 mm" ปลอมขึ้นมาตรงนั้น
      • erosion = ค่าต่ำสุดในหน้าต่าง จึงเอนไปทาง noise ด้านต่ำเสมอ (bias ≈ −3.8σ)

    ⚠ และห้ามใช้ base = (opening+closing)/2 — ฐานแบบนั้นจมลงไปครึ่งหลุม ทำให้ความลึก
      ที่วัดได้เหลือครึ่งเดียวเสมอ (พิสูจน์กับ synth: หลุมจริง 9.0 mm วัดได้ 4.8 mm)
    """
    v = dem.valid
    dev_plane = np.where(v, dem.z * 1000.0, np.nan).astype(np.float32)
    if not v.any() or not _HAS_CV2:
        nan = np.full(dem.shape, np.nan, np.float32)
        return dev_plane, (dev_plane.copy() if not _HAS_CV2 else nan), nan

    zf = _fill_nan_nearest(dem.z * 1000.0, v)
    ksz = max(3, int(round(p.base_win_m / dem.cell)) | 1)
    ksz = min(ksz, (min(dem.shape) - 1) | 1)
    ksz = max(ksz, 3)

    # ── รอบ 1: top-hat หาผู้ต้องสงสัย ──
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    opened = cv2.morphologyEx(zf, cv2.MORPH_OPEN, se, borderType=cv2.BORDER_REPLICATE)
    closed = cv2.morphologyEx(zf, cv2.MORPH_CLOSE, se, borderType=cv2.BORDER_REPLICATE)
    dev0 = (zf - opened) - (closed - zf)

    nz = np.where(np.isfinite(noise_mm), noise_mm, 0.5) if noise_mm is not None \
        else np.full(dem.shape, 0.5, np.float32)
    # เกณฑ์ผู้ต้องสงสัยตั้งหลวมกว่าเกณฑ์ตัดสินจริง (÷2) — ยอมตัดพื้นดีทิ้งบ้าง ดีกว่าปล่อย
    # ขอบหลุมหลุดเข้าไปปนในฐาน
    susp = np.abs(dev0) > np.maximum(0.5 * min(p.pothole_mm, p.bump_mm), 2.0 * nz)
    susp = cv2.dilate(susp.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)

    # ── รอบ 2: ฐาน = ค่าเฉลี่ยเฉพาะที่ของ "เซลล์ที่ไม่ต้องสงสัย" ──
    m = (v & ~susp).astype(np.float32)
    k = (ksz, ksz)
    s = cv2.blur(zf * m, k, borderType=cv2.BORDER_REPLICATE)
    w = cv2.blur(m, k, borderType=cv2.BORDER_REPLICATE)
    enough = w > 0.05                      # ต้องมีพื้นดีอย่างน้อย 5% ของหน้าต่าง
    base = np.where(enough, s / np.maximum(w, 1e-6), zf - dev0).astype(np.float32)

    dev_local = np.where(v, zf - base, np.nan).astype(np.float32)
    return dev_plane, dev_local, np.where(v, base, np.nan).astype(np.float32)


def local_roughness_mm(dev_local_mm: np.ndarray, cell: float,
                       win_m: float = 0.15) -> np.ndarray:
    """ความขรุขระประจำถิ่นของผิว (mm) — "ผิวนี้สั่นเองเท่าไรในสเกลเล็กกว่าล้อ"

    ทำไมต้องมีทั้งตัวนี้และ local_noise_mm:

      local_noise_mm วัดจากเศษเหลือของ median 3x3 (หน้าต่าง 30 mm) ซึ่งใช้ได้ดีกับ
      noise ที่สุ่มเป็นอิสระต่อพิกเซล แต่ depth จริงจาก D455 ผ่าน block matching มาแล้ว
      พิกเซลข้างกันจึงสัมพันธ์กันในระยะใกล้เคียงกับหน้าต่างที่ใช้วัด ตัวประมาณเลย
      "มองไม่เห็น" noise ของตัวเอง — วัดได้บนพรมจริง: ตัวเก่าให้ 0.3 mm คงที่ทุกระยะ
      ขณะที่ค่าเบี่ยงเบนจริงโตจาก 1.9 mm ที่ 0.4 m เป็น 8 mm ที่ 2 m

      ตัวนี้วัดในหน้าต่าง 200 mm ซึ่งกว้างกว่าความยาวสหสัมพันธ์ จึงจับได้ทั้ง noise
      ที่เกลี่ยกันมาแล้ว และ "เนื้อผิวจริง" (ขนพรม รอยต่อกระเบื้อง ผิวคอนกรีตขัด)
      สองอย่างนี้เหมือนกันในแง่ที่สำคัญ: ล้อกลิ้งข้ามไปเฉย ๆ ไม่ใช่หลุมที่ต้องซ่อม

    หาแบบสองรอบเพื่อไม่ให้หลุมจริงยกเกณฑ์ของตัวเอง: รอบแรกได้ค่าคร่าว ๆ
    รอบสองเฉลี่ยเฉพาะเซลล์ที่ไม่เกิน 3 เท่าของรอบแรก
    """
    if not _HAS_CV2 or dev_local_mm is None:
        return np.full_like(dev_local_mm, np.nan) if dev_local_mm is not None else None
    v = np.isfinite(dev_local_mm)
    if not v.any():
        return np.full(dev_local_mm.shape, np.nan, np.float32)
    x = np.where(v, dev_local_mm, 0.0).astype(np.float32)
    k = max(3, int(round(win_m / cell)) | 1)
    k = min(k, (min(dev_local_mm.shape) - 1) | 1)
    k = max(k, 3)

    vf = v.astype(np.float32)

    def box(z):
        return cv2.blur(z, (k, k), borderType=cv2.BORDER_REPLICATE)

    # กระจายรอบ "ค่ากลางของหน้าต่างเดียวกัน" ไม่ใช่รอบ median 5x5 —
    # ถ้าหักด้วย median แคบ ๆ จะเหลือแต่สเกลเล็กกว่า 50 mm ซึ่งจับลายพรม/รอยต่อ
    # กระเบื้องที่กว้าง 50-200 mm ไม่ได้เลย ทั้งที่นั่นคือสเกลที่ล้อข้ามไปเฉย ๆ
    w = box(vf)
    centre = np.where(w > 0.15, box(x * vf) / np.maximum(w, 1e-6), 0.0)
    ad = np.abs(x - centre)

    w0 = box(vf)
    r0 = np.where(w0 > 0.15, box(ad * vf) / np.maximum(w0, 1e-6), np.nan)
    keep = (v & (ad <= 3.0 * np.where(np.isfinite(r0), r0, ad))).astype(np.float32)
    w1 = box(keep)
    r1 = np.where(w1 > 0.15, box(ad * keep) / np.maximum(w1, 1e-6), r0)
    return (r1 * 1.4826).astype(np.float32)        # MAD -> sigma


def usable_mask(dem: DEM, noise_mm: np.ndarray, rough_mm: Optional[np.ndarray],
                p: SurfaceParams) -> np.ndarray:
    """เซลล์ที่ยัง "วัดหลุมได้จริง" ตามเกณฑ์ที่ตั้งไว้

    ที่ระยะไกล เกณฑ์ตรวจจับโตจนเกินขนาดหลุมที่สนใจ — ตรงนั้นไม่ใช่ "พื้นเรียบ"
    แต่คือ "มองไม่เห็นแล้ว" ถ้ายังนับรวมเป็นพื้นที่สำรวจ ตัวเลขพื้นที่กับ FF จะสวย
    เกินจริง และ "เล็กสุดที่เชื่อได้" จะถูกถ่วงด้วยบริเวณที่ไม่มีใครควรเชื่ออยู่แล้ว
    (วัดจริงบนพรม: 3.17 m² -> เล็กสุด 25 mm ทั้งที่ครึ่งหน้าใกล้ทำได้ถึง 4 mm)

    เอาเฉพาะก้อนที่ต่อเนื่องกับแถวหน้าสุด เพื่อไม่ให้หย่อมดี ๆ ที่โผล่หลังสิ่งกีดขวาง
    (ซึ่งกล้องมองไม่ถึงจริง) ถูกนับเข้ามา
    """
    thr = p.k_sigma * np.where(np.isfinite(noise_mm), noise_mm, 0.0)
    if rough_mm is not None:
        thr = np.maximum(thr, p.k_rough * np.where(np.isfinite(rough_mm), rough_mm, 0.0))
    ok = dem.valid & (thr <= p.max_usable_mm)
    if p.min_points_cell > 0 and _HAS_CV2:
        # เฉลี่ยในหน้าต่างเดียวกับ solid_mask เพื่อไม่ให้เซลล์เดี่ยวที่บังเอิญโหว่
        # เจาะรูพรุนไปทั้งแผ่น — เราสนใจ "ทั้งหย่อมข้อมูลบาง" ไม่ใช่เซลล์เดี่ยว
        kk = max(3, int(round(p.solid_win_m / dem.cell)) | 1)
        kk = min(kk, (min(dem.shape) - 1) | 1)
        if kk >= 3:
            dens = cv2.blur(dem.count.astype(np.float32), (kk, kk),
                            borderType=cv2.BORDER_REPLICATE)
            ok &= dens >= p.min_points_cell
    if not ok.any() or not _HAS_CV2:
        return ok
    n, lab = cv2.connectedComponents(ok.astype(np.uint8), connectivity=8)
    if n <= 2:
        return ok
    ids, cnt = np.unique(lab[lab > 0], return_counts=True)
    if ids.size == 0:
        return ok
    # เก็บก้อนใหญ่สุด + ก้อนที่ใหญ่เกิน 25% ของมัน
    #
    # เดิมยึด "แถวล่างสุดของ DEM" เป็นสมอ ซึ่งเปราะมาก: เฟรมไหนแถวล่างบังเอิญหลุด
    # เกณฑ์ไป พื้นที่สำรวจจะหล่นจาก 3.5 m² เหลือ 0.08 m² ทันทีทั้งที่ภาพแทบไม่เปลี่ยน
    biggest = int(cnt.max())
    keep = ids[cnt >= max(1, int(0.25 * biggest))]
    return np.isin(lab, keep)


def dem_range_m(dem: DEM, T_floor_cam: np.ndarray) -> np.ndarray:
    """ระยะจากกล้องถึงกึ่งกลางแต่ละเซลล์ (m) — ตัวแปรที่ noise ของ stereo ขึ้นกับมัน"""
    cam = (np.linalg.inv(T_floor_cam) @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    H, W = dem.shape
    xs = dem.x_max - (np.arange(H, dtype=np.float32) + 0.5) * dem.cell
    ys = dem.y_max - (np.arange(W, dtype=np.float32) + 0.5) * dem.cell
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    z = np.where(np.isfinite(dem.z), dem.z, 0.0)
    return np.sqrt((X - cam[0]) ** 2 + (Y - cam[1]) ** 2
                   + (z - cam[2]) ** 2).astype(np.float32)


def fit_texture_ab(rough_raw: np.ndarray, rng_m: np.ndarray,
                   valid: np.ndarray) -> Optional[Tuple[float, float]]:
    """ปรับเส้น rough(ระยะ) = a + b·ระยะ²  คืน (a, b) หรือ None ถ้าข้อมูลไม่พอ

    a = เนื้อผิวของวัสดุ (mm)   b = สัมประสิทธิ์ noise ของ stereo (mm ต่อ m²)

    แบ่งชั้นระยะด้วย "ควอนไทล์ของระยะ" ไม่ใช่ช่วงเท่า ๆ กัน — ถ้าใช้ linspace บนช่วง
    min..max เซลล์ไกลไม่กี่เซลล์จะยืดช่วงจนชั้นท้าย ๆ แทบว่าง แล้ว fit เอียงตามมันทั้งเส้น
    (อาการที่วัดได้: เนื้อผิวเด้ง 4 -> 32 mm ระหว่างเฟรมที่ข้อมูลนิ่งสนิท)
    """
    m = valid & np.isfinite(rough_raw) & np.isfinite(rng_m)
    if m.sum() < 400:
        return None
    r, g = rough_raw[m], rng_m[m]
    if float(g.max() - g.min()) < 0.10:
        return (float(np.percentile(r, 25)), 0.0)

    edges = np.percentile(g, np.linspace(0, 100, 11))
    edges = np.unique(edges)
    if edges.size < 4:
        return (float(np.percentile(r, 25)), 0.0)

    xs, ys, ws = [], [], []
    for i in range(edges.size - 1):
        sel = (g >= edges[i]) & (g <= edges[i + 1] if i == edges.size - 2
                                 else g < edges[i + 1])
        c = int(sel.sum())
        if c < 200:                       # ชั้นบางเกินไป ตัดทิ้ง อย่าให้ลาก fit
            continue
        xs.append(float(np.median(g[sel])))
        ys.append(float(np.percentile(r[sel], 25)))
        ws.append(c)
    if len(xs) < 3:
        return (float(np.percentile(r, 25)), 0.0)

    w = np.sqrt(np.asarray(ws, float))
    A = np.stack([np.ones(len(xs)), np.square(xs)], axis=1) * w[:, None]
    coef, *_ = np.linalg.lstsq(A, np.asarray(ys) * w, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    # ขอบเขตความเป็นไปได้ทางกายภาพ: b ของ D455 ตามทฤษฎีราว 1.2 mm/m²
    # ของจริงแย่กว่านั้นได้หลายเท่า แต่ไม่ถึงหลักร้อย — กันไม่ให้ fit หลุดโลก
    b = float(np.clip(b, 0.0, 60.0))
    a = float(np.clip(a, 0.0, 30.0))
    return (a, b)


def surface_floor_mm(rough_raw: np.ndarray, rng_m: np.ndarray, valid: np.ndarray,
                     prior_ab: Optional[Tuple[float, float]] = None,
                     alpha: float = 0.30):
    """คืน (แผนที่เนื้อผิวต่อเซลล์ mm, (a, b) ที่ใช้จริง)

    prior_ab = ค่าจากเฟรมก่อน ถ้าส่งมาจะเกลี่ยแบบ EMA: ใหม่ = alpha·fit + (1-alpha)·prior

    ทำไมต้องเกลี่ยข้ามเฟรม: เกณฑ์ตรวจจับโตตามระยะกำลังสอง ความคลาดของ b เพียงเล็กน้อย
    จึงถูกขยายเป็นการขยับขอบเขต "บริเวณที่วัดได้" อย่างมโหฬาร วัดจริงตอนยังไม่เกลี่ย:
    ข้อมูลเข้านิ่งสนิท (fill sd 0.00, rms sd 0.08) แต่พื้นที่สำรวจเด้ง 0.08-3.51 m²
    และจำนวนจุดที่รายงานเด้ง 0-26 ในภาพนิ่ง ๆ ภาพเดียว
    """
    ab = fit_texture_ab(rough_raw, rng_m, valid)
    if ab is None:
        ab = prior_ab
    elif prior_ab is not None:
        a0, b0 = prior_ab
        ab = (alpha * ab[0] + (1.0 - alpha) * a0,
              alpha * ab[1] + (1.0 - alpha) * b0)
    if ab is None:
        base = np.where(np.isfinite(rough_raw), rough_raw, 0.0).astype(np.float32)
        return base, None
    a, b = ab
    g = np.where(np.isfinite(rng_m), rng_m, 0.0)
    return (a + b * np.square(g)).astype(np.float32), ab


def _shift(a: np.ndarray, dr: int, dc: int, fill=np.nan) -> np.ndarray:
    out = np.full_like(a, fill)
    H, W = a.shape
    sr = slice(max(0, dr), H + min(0, dr))
    sc = slice(max(0, dc), W + min(0, dc))
    tr = slice(max(0, -dr), H + min(0, -dr))
    tc = slice(max(0, -dc), W + min(0, -dc))
    out[sr, sc] = a[tr, tc]
    return out


def step_map(dem: DEM, base_m: float = 0.04) -> np.ndarray:
    """ต่างระดับสูงสุดข้ามจุดนั้น วัดบนฐานยาว base_m (mm) — จับ "ลิ่ม/ขอบรอยต่อ"

    ต่างจาก dev_local ตรงที่ลิ่ม 3 mm ขอบคม กับแอ่งลาด 3 mm กว้าง 30 cm ให้ค่า dev
    เท่ากันแต่อันตรายคนละเรื่อง — อันแรกกระแทกล้อ caster จนแตก (iscano) อันหลังแทบไม่รู้สึก

    วัดจาก "สองฝั่งห่างกัน base_m" ไม่ใช่เซลล์ที่ติดกัน เพราะผลต่างเซลล์ติดกันคือเครื่องขยาย
    noise ชั้นดี (σ_diff = √2·σ) — ฐานยาวขึ้นบวก median ก่อนหน้า กด noise ลงราว 3 เท่า
    โดยขอบคมยังคมอยู่ (ขอบจริงกว้างกว่า 1 เซลล์เสมอเมื่อมองผ่าน DEM)
    """
    z = np.where(dem.valid, dem.z * 1000.0, np.nan).astype(np.float32)
    n = max(1, int(round(base_m / dem.cell / 2.0)))     # ครึ่งฐานเป็นจำนวนเซลล์
    out = np.zeros(dem.shape, np.float32)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = _shift(z, dr * n, dc * n)
        b = _shift(z, -dr * n, -dc * n)
        diff = np.abs(a - b)
        good = np.isfinite(diff)
        out[good] = np.maximum(out[good], diff[good])
    out[~dem.valid] = np.nan
    return out


def slope_map(dem: DEM, window_m: float = 0.20) -> np.ndarray:
    """ความชันเฉพาะที่ (องศา) วัดบนฐานยาว window_m — ใช้เทียบกับมุมไต่สูงสุดของหุ่น"""
    if not _HAS_CV2:
        return np.zeros(dem.shape, np.float32)
    v = dem.valid
    zf = _fill_nan_nearest(dem.z, v)
    k = max(3, int(round(window_m / dem.cell)) | 1)
    k = min(k, 31)                                    # Sobel รองรับถึง 31
    gx = cv2.Sobel(zf, cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(zf, cv2.CV_32F, 0, 1, ksize=k)
    # normalise ค่าคูณของ Sobel kernel ให้เป็นความชันจริง (m/m)
    # ตัวหารที่ถูกคือ "โมเมนต์อันดับหนึ่ง" ของเคอร์เนลอนุพันธ์ Σ|k[j]·j| คูณผลรวมเคอร์เนลเรียบ
    # (ใช้ Σ|k[j]| เฉย ๆ จะผิด — วัดจริงเพี้ยนไป 2.84 เท่าที่ ksize=21)
    kd, ks = cv2.getDerivKernels(1, 0, k)
    j = np.arange(k) - k // 2
    norm = float(abs((kd.ravel() * j).sum()) * np.abs(ks.ravel()).sum()) * dem.cell
    if norm <= 0:
        norm = 1.0
    g = np.sqrt(gx * gx + gy * gy) / norm
    out = np.degrees(np.arctan(g)).astype(np.float32)
    out[~v] = np.nan
    return out


# ───────────────────────────── ตรวจจับ defect ─────────────────────────────

def _components(mask: np.ndarray):
    """คืน (labels, n) — ใช้ cv2 ถ้ามี ไม่งั้น fallback แบบ flood fill"""
    if _HAS_CV2:
        n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return lab, n - 1
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    return lab, n


# อคติที่เหลือของ p97 หลังหักด้วย noise — คาลิเบรตจากการกวาด synth (ดู evaluate.run_capability_matrix)
_PEAK_DEBIAS = 1.881          # Φ⁻¹(0.97); ปรับตามผลวัดจริงถ้าจำเป็น


def _blobs_to_defects(lab, n, dev_mm, dem: DEM, kind: str, p: SurfaceParams, sign: int,
                      noise_mm: np.ndarray, steps_mm: Optional[np.ndarray],
                      cam_height_m: float,
                      mean_src: Optional[np.ndarray] = None,
                      peak_src: Optional[np.ndarray] = None) -> List[Defect]:
    out = []
    min_cells = max(3, int(round(p.min_defect_area_m2 / (dem.cell ** 2))))
    for i in range(1, n + 1):
        m = lab == i
        cells = int(m.sum())
        if cells < min_cells:
            continue
        rr, cc = np.nonzero(m)
        r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
        # peak_src ให้วัดความสูง/ลึกจากแผนที่อื่นได้ โดยยังใช้ dev_mm เป็นตัวแบ่งก้อน
        # (สิ่งกีดขวางวัดจากระนาบพื้น ไม่ใช่จากฐานเฉพาะที่ — ดูเหตุผลใน detect_defects)
        src = dev_mm if peak_src is None else peak_src
        vals = src[m]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        nz = noise_mm[m]
        nz = nz[np.isfinite(nz)]
        noise = float(np.median(nz)) if nz.size else 0.0
        # ใช้เปอร์เซ็นไทล์ ไม่ใช่ max: max ของ N เซลล์ที่มี noise จะสูงเกินจริงเสมอ
        # แล้วหักอคติที่เหลือของ p97 ออกด้วย noise ที่วัดได้ (Φ⁻¹(0.97) = 1.881)
        peak = float(np.percentile(vals, 3.0 if sign < 0 else 97.0))
        peak -= sign * _PEAK_DEBIAS * noise
        mvals = vals if mean_src is None else mean_src[m][np.isfinite(mean_src[m])]
        snr = abs(peak) / noise if noise > 1e-6 else float('inf')
        if snr < p.min_snr:
            continue                       # อ่อนเกินกว่าจะแยกจาก noise ได้ — ไม่ยืนยัน
        # ─ ด่านเชิงฟิสิกส์: ล้อจะ "รู้สึก" กับสิ่งนี้จริงไหม ─
        # ร่องแคบกว่าล้อจะถูกล้อคร่อมข้าม ล้อจมลงไปได้แค่
        #     δ = R − √(R² − (w/2)²)
        # เนื้อผิวพรม/ผิวคอนกรีตหยาบให้จุดกว้าง 20-70 mm ซึ่งล้อ Ø75 คร่อมได้เกือบหมด
        # (กว้าง 30 mm → จมได้ 3.1 mm ไม่ว่าร่องจะลึกแค่ไหน)
        # ไม่ใช้กับ step เพราะลิ่มขอบคมแคบ ๆ กระแทกล้อได้จริง
        if p.min_effective_mm > 0 and kind in ('pothole', 'bump'):
            w_m = min((r1 - r0 + 1), (c1 - c0 + 1)) * dem.cell   # ด้านแคบสุดคือด้านที่ล้อคร่อม
            R = max(p.wheel_radius_m, 1e-3)
            bridge_mm = (1e3 * (R - np.sqrt(max(R * R - (w_m / 2) ** 2, 0.0)))
                         if w_m < 2 * R else abs(peak))
            if min(abs(peak), bridge_mm) < p.min_effective_mm:
                continue

        x, y = dem.xy_of(rr.mean(), cc.mean())
        edge = 0.0
        if steps_mm is not None:
            sv = steps_mm[m]
            sv = sv[np.isfinite(sv)]
            if sv.size:
                edge = float(sv.max())
        out.append(Defect(
            kind=kind, x=float(x), y=float(y),
            length_m=float((r1 - r0 + 1) * dem.cell),
            width_m=float((c1 - c0 + 1) * dem.cell),
            area_m2=float(cells * dem.cell ** 2),
            peak_mm=peak, mean_mm=float(mvals.mean()) if len(mvals) else float('nan'),
            cells=cells, bbox=(int(r0), int(c0), int(r1), int(c1)),
            edge_step_mm=edge, noise_mm=noise, snr=float(snr),
            range_m=float(np.hypot(x, cam_height_m))))
    return out


def _bbox_overlap_frac(a, b) -> float:
    r0, c0, r1, c1 = a
    kr0, kc0, kr1, kc1 = b
    ir = max(0, min(r1, kr1) - max(r0, kr0) + 1)
    ic = max(0, min(c1, kc1) - max(c0, kc0) + 1)
    inter = ir * ic
    if inter <= 0:
        return 0.0
    a1 = (r1 - r0 + 1) * (c1 - c0 + 1)
    a2 = (kr1 - kr0 + 1) * (kc1 - kc0 + 1)
    return inter / float(min(a1, a2))


def detect_defects(dem: DEM, dev_local_mm: np.ndarray, steps_mm: np.ndarray,
                   noise_mm: np.ndarray, p: SurfaceParams,
                   cam_height_m: float = 0.0,
                   solid: Optional[np.ndarray] = None,
                   rough_mm: Optional[np.ndarray] = None,
                   dev_plane_mm: Optional[np.ndarray] = None) -> List[Defect]:
    """แยกเป็น 5 ชนิด: หลุม / เนิน / ลิ่ม(step) / สิ่งกีดขวาง / รูข้อมูล

    ลำดับสำคัญ: ตรวจ "พื้นที่" (หลุม/เนิน) ก่อน แล้วค่อยตรวจ step — step ที่ทับกับหลุมอยู่แล้ว
    ไม่ใช่ defect ใหม่ แต่คือ "ขอบของหลุมนั้น" จึงเก็บเป็น edge_step_mm ของหลุมแทน
    (ก่อนแก้: หลุมขอบชันถูกรายงานเป็น 'step' ทำให้ชนิดผิดและนับซ้ำ)

    รูข้อมูล (missing) สำคัญกว่าที่คิด: หลุมลึกหรือผิวดำมัน ทำให้ stereo หา correspondence
    ไม่ได้ → depth หลุดเป็นรู "ไม่มีข้อมูล" ไม่ใช่ "พื้นเรียบ" ต้องรายงานแยก ไม่ใช่เงียบไป
    """
    v = dem.valid
    d = np.where(v, dev_local_mm, 0.0)
    nz = np.where(np.isfinite(noise_mm), noise_mm, 1.0)
    st = np.where(np.isfinite(steps_mm), steps_mm, 0.0) if steps_mm is not None else None

    # threshold ต่อเซลล์ = max(ค่าคงที่, k·noise, k·ความขรุขระของผิวตรงนั้น)
    #
    # เทอมที่สามคือตัวที่ทำให้ใช้กับพรม/กระเบื้องมีลายได้จริง: บนพรม เนื้อผิวเองสั่น
    # 2-5 mm อยู่แล้ว ถ้าตั้งเกณฑ์ตายตัว 4 mm จะได้ "หลุม" เป็นร้อยจุดซึ่งล้อไม่รู้สึกเลย
    # และมันยังคุมระยะไกลไปในตัว เพราะความขรุขระที่วัดได้โตตามระยะพอดี (0.6 mm ที่ 0.4 m
    # -> 7.7 mm ที่ 2 m) จึงไม่ต้องตั้งระยะตัดตายตัวให้เดาผิดเดาถูก
    rg = (np.where(np.isfinite(rough_mm), rough_mm, 0.0)
          if rough_mm is not None else np.zeros_like(nz))
    floor_mm = np.maximum(p.k_sigma * nz, p.k_rough * rg)
    thr_hole = np.maximum(p.pothole_mm, floor_mm)
    thr_bump = np.maximum(p.bump_mm, floor_mm)
    thr_step = np.maximum(p.step_mm, floor_mm * np.sqrt(2.0))   # ผลต่างสองฝั่ง → √2·σ

    # ความสูงเหนือ "ระนาบพื้น" (ไม่ผ่านฐานเฉพาะที่) — ใช้กับสิ่งกีดขวางเท่านั้น
    #
    # หลุม/เนิน คือรูปร่างของตัวพื้นเอง ต้องตัดความลาดเอียงคลื่นยาวออกด้วยฐานเฉพาะที่
    # แต่ "ของที่วางอยู่บนพื้น" สิ่งที่มีความหมายคือมันสูงจากพื้นเท่าไร และฐานเฉพาะที่
    # (base_win_m 0.4 m) จะปีนขึ้นไปอยู่บนตัวของเองเมื่อของกว้างใกล้ขนาดหน้าต่าง
    # ทำให้ความสูงที่รายงานยุบลงและแกว่ง วัดจริงกับกล่องสูง 355 mm 8 เฟรมติด:
    #     จากฐานเฉพาะที่   71-112 mm  (sd 13)
    #     จากระนาบพื้น    353-356 mm  (sd  1)
    hgt = np.where(v, dev_plane_mm, 0.0) if dev_plane_mm is not None else d

    defects: List[Defect] = []
    hole_mask = v & (d <= -thr_hole)
    # นับเป็นสิ่งกีดขวางถ้าสูงจากพื้นถึงเกณฑ์ หรือโผล่จากฐานเฉพาะที่ถึงเกณฑ์ — อย่างใด
    # อย่างหนึ่ง เพราะกล่องใบเดียวเคยถูกซอยเป็น obstacle 71 mm + bump 52 mm ทั้งที่
    # ทั้งสองก้อนสูงจากพื้น 355 mm เท่ากัน (ตรงกลางกล่อง dev_local ยุบเหลือเกือบศูนย์)
    obst_mask = v & ((hgt >= p.obstacle_mm) | (d >= p.obstacle_mm))
    bump_mask = v & (d >= thr_bump) & ~obst_mask
    for mask, kind, sign in ((hole_mask, 'pothole', -1),
                             (bump_mask, 'bump', +1),
                             (obst_mask, 'obstacle', +1)):
        if mask.any():
            lab, n = _components(mask)
            defects += _blobs_to_defects(
                lab, n, d, dem, kind, p, sign, nz, st, cam_height_m,
                peak_src=(hgt if kind == 'obstacle' and dev_plane_mm is not None
                          else None))

    # ลิ่ม/ขอบคมที่ "ยืนอยู่ลำพัง" (รอยต่อพื้น ขอบแผ่น) — ที่ทับ defect เดิมถือเป็นขอบของมัน
    if st is not None:
        step_mask = v & (st >= thr_step)
        if step_mask.any():
            lab, n = _components(step_mask)
            for s in _blobs_to_defects(lab, n, st, dem, 'step', p, +1, nz, st,
                                       cam_height_m, mean_src=d):
                if all(_bbox_overlap_frac(s.bbox, e.bbox) < 0.35 for e in defects):
                    defects.append(s)

    # รูข้อมูลที่ล้อมรอบด้วยพื้น (ไม่ใช่ขอบกริด และไม่ใช่บริเวณที่จุดโปร่งอยู่แล้ว)
    holes = (~v).astype(np.uint8)
    if holes.any():
        lab, n = _components(holes)
        min_cells = max(2, int(round(p.missing_min_area_m2 / (dem.cell ** 2))))
        H, W = dem.shape
        for i in range(1, n + 1):
            m = lab == i
            cells = int(m.sum())
            if cells < min_cells:
                continue
            rr, cc = np.nonzero(m)
            if rr.min() == 0 or cc.min() == 0 or rr.max() == H - 1 or cc.max() == W - 1:
                continue                      # ติดขอบกริด = แค่มองไม่ถึง ไม่ใช่ความผิดปกติ
            # รูจริงต้องอยู่ "กลางบริเวณข้อมูลแน่น" ไม่ใช่ตรงที่จุดโปร่งอยู่แล้วเป็นปกติ
            # ไม่งั้นได้ false alarm เพียบตรงขอบไกลของกริด (เจอจริงตอนทดสอบกับ synth)
            if solid is not None:
                pad = 5
                rs = slice(max(0, rr.min() - pad), min(H, rr.max() + pad + 1))
                cs = slice(max(0, cc.min() - pad), min(W, cc.max() + pad + 1))
                ring = solid[rs, cs] | m[rs, cs]
                if ring.size == 0 or ring.mean() < 0.85:
                    continue
            r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
            x, y = dem.xy_of(rr.mean(), cc.mean())
            defects.append(Defect(
                kind='missing', x=float(x), y=float(y),
                length_m=float((r1 - r0 + 1) * dem.cell),
                width_m=float((c1 - c0 + 1) * dem.cell),
                area_m2=float(cells * dem.cell ** 2),
                peak_mm=float('nan'), mean_mm=float('nan'), cells=cells,
                bbox=(int(r0), int(c0), int(r1), int(c1)),
                range_m=float(np.hypot(x, cam_height_m)),
                reason='ไม่มีข้อมูล depth — อาจเป็นหลุมลึก ผิวสะท้อน/ดำมัน หรือโดนบัง'))
    return defects


def merge_overlapping(defects: List[Defect], iou_thr: float = 0.5) -> List[Defect]:
    """ยุบ defect ที่ทับกันมาก — เก็บอันที่เบี่ยงเบนแรงกว่าไว้"""
    def sev(d: Defect):
        return abs(d.peak_mm) if np.isfinite(d.peak_mm) else 0.0

    kept: List[Defect] = []
    for d in sorted(defects, key=sev, reverse=True):
        if all(_bbox_overlap_frac(d.bbox, k.bbox) < iou_thr for k in kept):
            kept.append(d)
    return kept


# ───────────────────────────── ท่อรวม ─────────────────────────────

@dataclass
class SurfaceResult:
    ok: bool
    plane: Plane
    T_floor_cam: Optional[np.ndarray] = None
    dem: Optional[DEM] = None
    dev_plane_mm: Optional[np.ndarray] = None
    dev_local_mm: Optional[np.ndarray] = None
    steps_mm: Optional[np.ndarray] = None
    slope_deg: Optional[np.ndarray] = None
    noise_mm: Optional[np.ndarray] = None
    rough_mm: Optional[np.ndarray] = None      # ความขรุขระของเนื้อผิวเอง (พรม/กระเบื้อง)
    texture_ab: Optional[Tuple[float, float]] = None      # (a, b) ที่ใช้ — ส่งกลับเข้าเฟรมถัดไป
    solid: Optional[np.ndarray] = None         # เซลล์ที่อยู่ในบริเวณข้อมูลแน่นจริง
    defects: List[Defect] = field(default_factory=list)
    pts_floor: Optional[np.ndarray] = None     # (H,W,3) ใน floor frame — ใช้วาด overlay
    valid: Optional[np.ndarray] = None
    note: str = ''

    @property
    def detectable_mm(self) -> float:
        """ขนาด defect ที่เล็กที่สุดที่ "ยืนยันได้" ในเฟรมนี้ (mm) — ค่ากลางทั้งภาพ
        เป็นตัวเลขที่ตอบตรง ๆ ว่า ณ ระยะ/สภาพนี้ กล้องเชื่อถือได้ถึงระดับไหน"""
        if self.noise_mm is None or self.dem is None:
            return float('nan')
        v = self.dem.valid & np.isfinite(self.noise_mm)
        if not v.any():
            return float('nan')
        lim = self.noise_mm[v] * 4.0
        if self.rough_mm is not None:
            # เนื้อผิวเองก็เป็นเพดานของสิ่งที่แยกออกได้ ไม่ใช่แค่ noise ของเซนเซอร์
            rg = np.where(np.isfinite(self.rough_mm), self.rough_mm, 0.0)[v]
            lim = np.maximum(lim, rg * 3.0)      # ต้องตรงกับ k_rough ที่ใช้ตรวจจับ
        return float(np.median(lim))


def _specular_note(depth_mm: np.ndarray, p: SurfaceParams) -> str:
    """จับลายเซ็นของ "ผิวกระจก" — คืนคำอธิบายถ้าเจอ ไม่เจอคืนสตริงว่าง

    บนผิวเคลือบเงา ลาย IR ที่กล้องยิงไปไม่กระเจิงกลับ แต่สะท้อนแบบกระจก
    สิ่งที่ stereo จับคู่ได้จึงเป็น "เงาของเพดาน/ห้องในผิวโต๊ะ" ไม่ใช่ผิวโต๊ะ
    ระยะที่ได้เลยกลายเป็น (ระยะถึงผิว + ระยะจากผิวถึงเพดาน) ซึ่งไกลเกินจริงมาก

    วัดจากโต๊ะไม้เคลือบเงาจริง: ได้ depth แค่ 16% ของภาพ และครึ่งหนึ่งของนั้น
    รายงานระยะเกิน 1.5 m (ไปถึง 65 m) ทั้งที่ผิวโต๊ะอยู่ห่าง 0.5-1.2 m

    แยกจาก "แค่แสงจ้า" ได้ชัดเจน: ถ้าเป็นเรื่องค่าแสง พิกเซลสว่างจะเสียมากกว่า
    พิกเซลมืด แต่เคสกระจกเสียเท่ากันหมดทุกระดับความสว่าง (วัดแล้ว 14.5-19.6%)
    """
    raw = depth_mm > 0
    n = int(raw.sum())
    if n < 200:
        return ''
    raw_fill = float(raw.mean())
    z = depth_mm[raw].astype(np.float32) * 0.001
    far_frac = float((z > p.zmax).mean())
    if raw_fill < 0.6 and far_frac > 0.30:
        return (f'ผิวสะท้อนแบบกระจก — ได้ค่า depth แค่ {raw_fill*100:.0f}% ของภาพ '
                f'และ {far_frac*100:.0f}% ของนั้นรายงานระยะเกิน {p.zmax:.1f} m '
                f'(ไกลสุด {z.max():.0f} m) แปลว่ากล้องกำลังวัด "เงาสะท้อนของห้อง" '
                'ในผิวนั้น ไม่ใช่ตัวผิว · ไม่มีค่ากล้องตัวไหนแก้ได้ '
                '(กวาดแล้ว laser 0-360 กับ exposure 4-200 ms ให้ผลเท่ากันหมด) '
                'ต้องเปลี่ยนผิว เช่น ปูกระดาษ/เทปด้าน หรือดับไฟที่สะท้อนอยู่เหนือผิว')
    return ''


def analyse_depth(depth_mm: np.ndarray, K, p: SurfaceParams,
                  plane: Optional[Plane] = None,
                  want_slope: bool = True) -> SurfaceResult:
    """ท่อเต็มจาก depth image → ผลวิเคราะห์พื้น

    plane: ถ้าส่งมา จะ "ล็อกระนาบอ้างอิง" ไว้ (โหมดวัดจริงบนหุ่นที่คาลิเบรตแล้ว)
           ถ้าไม่ส่ง จะ fit ใหม่ทุกเฟรม (โหมดถือกล้องเดินสำรวจ)
    """
    st = max(1, int(p.pixel_stride))
    if st > 1:
        # ย่อภาพพร้อม "ย่อ K ให้ตรงกัน" — สูตรมาตรฐานของการ subsample:
        # พิกเซลใหม่ i ครอบคลุมพิกเซลเดิม i*st จึงต้องเลื่อนจุดกึ่งกลางด้วย 0.5
        depth_mm = depth_mm[::st, ::st]
        fx, fy, cx, cy = K
        K = (fx / st, fy / st, (cx + 0.5) / st - 0.5, (cy + 0.5) / st - 0.5)
    xyz, valid = deproject_full(depth_mm, K, p.zmin, p.zmax)
    fill = float(valid.mean())
    spec_note = _specular_note(depth_mm, p)
    if spec_note:
        # ปฏิเสธตั้งแต่ต้น ไม่ต้องรอให้ด่านอื่นจับ: ถ้าค่าที่ได้กลับมาส่วนใหญ่เป็นระยะ
        # ที่เป็นไปไม่ได้ ส่วนที่ "ดูดี" ก็เชื่อไม่ได้เหมือนกัน — มันคือเงาสะท้อนที่
        # บังเอิญตกในช่วงระยะที่รับ ไม่ใช่ผิวจริง
        return SurfaceResult(ok=False, plane=Plane(np.array([0., -1., 0.]), 0.0),
                             valid=valid, note=spec_note)
    if valid.sum() < 500:
        return SurfaceResult(
            ok=False, plane=Plane(np.array([0., -1., 0.]), 0.0), valid=valid,
            note=spec_note or
                 (f'แทบไม่มีค่า depth เลย (ได้ {fill*100:.0f}% ของภาพ) — ผิวสะท้อนแสง '
                  f'ผิวดำเงา หรือกระจก ทำให้ stereo จับคู่ภาพไม่ได้ '
                  f'ลองเปิด HDR: ros2 param set /camera/camera depth_module.hdr_enabled true'))

    if plane is None:
        sub = xyz[::p.stride_fit, ::p.stride_fit][valid[::p.stride_fit, ::p.stride_fit]]
        plane = fit_reference_plane(sub, p)
        if not plane.ok:
            # แยกสองสาเหตุที่แก้คนละทางกันให้ชัด ไม่งั้นผู้ใช้ไปนั่งขยับกล้องทั้งที่
            # ปัญหาคือแสงสะท้อนทำให้ depth เป็นรู
            if fill < 0.45:
                why = (f'ได้ค่า depth แค่ {fill*100:.0f}% ของภาพ — ผิวมันวาว/สะท้อนแสง '
                       'ทำให้เป็นรูโหว่ ที่เหลือเลยไม่พอ fit ระนาบ '
                       'ลองเปิด HDR หรือขยับมุมกล้องให้พ้นเงาสะท้อน '
                       '(python3 -m floor_inspect.tune_camera --apply)')
            else:
                why = (f'ได้ depth มา {fill*100:.0f}% ของภาพ แต่กระจายกันคนละทิศ '
                       'ในเฟรมน่าจะมีหลายระนาบปนกัน (พื้น+ผนัง+ของกอง) '
                       'ลองก้มกล้องลงให้พื้นเต็มเฟรมกว่านี้')
            return SurfaceResult(
                ok=False, plane=plane, valid=valid,
                note=spec_note or
                     (f'หาระนาบพื้นไม่เจอ — จุดที่แนบระนาบมีแค่ {plane.inlier_frac*100:.0f}% '
                      f'(ต้องการ ≥{p.min_inlier_frac*100:.0f}%) · {why}'))
        # ด่านกันรายงานขยะ: ถ้าระนาบที่ fit ได้ "ไม่เรียบ" ตั้งแต่ต้น แปลว่าสิ่งที่จับได้
        # ไม่ใช่พื้น (เป็นผนัง กองของ หรือฉากทั้งห้อง) — ค่าเบี่ยงเบนที่คำนวณต่อจะไร้ความหมาย
        # และจะพ่นรายการ "หลุมลึก 800 mm" ออกมาเป็นสิบจุด ซึ่งอันตรายกว่าการบอกว่าวัดไม่ได้
        if np.isfinite(plane.rms_mm) and plane.rms_mm > p.max_plane_rms_mm:
            return SurfaceResult(
                ok=False, plane=plane, valid=valid,
                note=spec_note or
                     (f'ระนาบที่จับได้ไม่เรียบพอจะเป็นพื้น (กระจาย {plane.rms_mm:.0f} mm '
                      f'เกินเกณฑ์ {p.max_plane_rms_mm:.0f} mm, depth ได้ {fill*100:.0f}%) — '
                      + ('ผิวมันวาวทำให้ stereo จับคู่ผิดจุด ค่าที่ได้เลยกระโดด '
                         'ลองเปิด HDR หรือขยับมุมกล้อง'
                         if fill < 0.6 else
                         'กล้องน่าจะยังไม่ได้เล็งพื้นราบ หรือมีของกองอยู่เต็มเฟรม')))

    T = floor_transform(plane)
    pts_floor = transform_points(xyz, T)
    dem = build_dem(pts_floor[valid], p)
    if dem is None:
        return SurfaceResult(ok=False, plane=plane, T_floor_cam=T, valid=valid,
                             pts_floor=pts_floor, note='สร้าง DEM ไม่ได้')
    dem = crop_to_surveyed(dem, p.solid_frac, p.solid_win_m)
    # ด่านที่สอง: เกณฑ์ rms อย่างเดียวไม่พอ — ผิวเรียบผืนจิ๋วใกล้เลนส์ (ฝากล่อง มือ โต๊ะ)
    # ก็ให้ rms ต่ำได้ แล้วระบบจะรายงาน "ตรวจแล้ว ไม่พบอะไร" ทั้งที่สำรวจได้ 0.04 m²
    # ซึ่งอันตรายกว่าการบอกว่ายังตรวจไม่ได้
    if dem.area_m2 < p.min_survey_m2:
        return SurfaceResult(
            ok=False, plane=plane, T_floor_cam=T, valid=valid, pts_floor=pts_floor,
            note=f'พื้นที่ที่สำรวจได้น้อยเกินไป ({dem.area_m2:.2f} m² '
                 f'ต่ำกว่าเกณฑ์ {p.min_survey_m2:.2f} m²) — กล้องน่าจะเล็งใกล้เกินไป '
                 'หรือมีของบังเกือบเต็มเฟรม')

    noise = local_noise_mm(dem, p.noise_win_m)     # ต้องวัดจาก DEM ดิบ ก่อน median filter
    dem_s = median_prefilter(dem, p.median_pre)
    dev_plane, dev_local, _ = deviation_maps(dem_s, p, noise)
    steps = step_map(dem_s, p.step_base_m)
    slope = slope_map(dem_s) if want_slope else None
    solid = solid_mask(dem_s, p.solid_win_m, p.solid_frac)
    r_local = local_roughness_mm(dev_local, dem_s.cell, p.rough_win_m)
    tex_ab = p.texture_prior
    rough = r_local
    rng_m = dem_range_m(dem_s, T)
    if r_local is not None:
        rough, tex_ab = surface_floor_mm(r_local, rng_m, dem_s.valid,
                                         prior_ab=p.texture_prior, alpha=p.texture_alpha)
        # เส้น fit เป็นค่า "ทั้งภาพ" จึงมองไม่เห็นว่าบางหย่อมคุณภาพ depth แย่กว่าที่อื่นมาก
        # (ห้องมืดลง มุมสะท้อน เงาของตัวเอง) — วัดจริง: fit ให้ p50 3.1 mm ขณะที่ค่าจริง
        # p50 4.1 p90 8.1 mm ผลคือหย่อมแย่ ๆ พ่นจุดปลอมออกมาเรื่อย ๆ
        #
        # opening เชิงสัณฐานคือเครื่องมือที่ตรงงาน: มันลบ "ยอดแคบ" ที่ defect สร้างขึ้นเอง
        # (หลุมกว้าง w ดัน r_local ขึ้นในรัศมี w + หน้าต่างวัด) แต่เก็บ "บริเวณกว้าง"
        # ที่แย่จริงไว้ทั้งผืน
        if _HAS_CV2 and p.rough_open_m > 0 and r_local is not None:
            k = max(3, int(round(p.rough_open_m / dem_s.cell)) | 1)
            k = min(k, (min(dem_s.shape) - 1) | 1)
            if k >= 3:
                se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                rf = np.where(np.isfinite(r_local), r_local, 0.0).astype(np.float32)
                op = cv2.morphologyEx(rf, cv2.MORPH_OPEN, se,
                                      borderType=cv2.BORDER_REPLICATE)
                rough = np.maximum(rough, op)

    # ตัดบริเวณที่เกณฑ์โตเกินจนวัดหลุมไม่ได้แล้วออกจาก "พื้นที่ที่สำรวจ"
    # (ทำก่อนตรวจ defect เพื่อให้พื้นที่/FF/เล็กสุดที่เชื่อได้ พูดถึงบริเวณเดียวกันหมด)
    use = usable_mask(dem_s, noise, rough, p)
    if use.any() and not use.all():
        z2 = np.where(use, dem_s.z, np.nan).astype(np.float32)
        dem_s = DEM(z=z2, count=np.where(use, dem_s.count, 0),
                    spread_mm=np.where(use, dem_s.spread_mm, np.nan).astype(np.float32),
                    cell=dem_s.cell, x_max=dem_s.x_max, y_max=dem_s.y_max)
        for arr in (dev_plane, dev_local, steps, noise, rough):
            if arr is not None:
                arr[~use] = np.nan
        if slope is not None:
            slope[~use] = np.nan
        solid = solid & use

    defects = merge_overlapping(
        detect_defects(dem_s, dev_local, steps, noise, p, plane.cam_height_m, solid,
                       rough_mm=rough, dev_plane_mm=dev_plane))
    return SurfaceResult(ok=True, plane=plane, T_floor_cam=T, dem=dem_s,
                         dev_plane_mm=dev_plane, dev_local_mm=dev_local,
                         steps_mm=steps, slope_deg=slope, noise_mm=noise,
                         rough_mm=rough, texture_ab=tex_ab, solid=solid,
                         defects=defects, pts_floor=pts_floor, valid=valid)
