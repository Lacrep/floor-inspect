# -*- coding: utf-8 -*-
"""synth — สร้าง "สนามทดสอบจำลอง" เป็น depth image เหมือนที่ D455 จะเห็นจริง (ROS-free)

ทำไมต้องมี: ตอนนี้ยังไม่มีพื้นที่จริงให้ทดลอง แต่การจะรู้ว่าอัลกอริทึม "วัดถูกจริงไหม"
ต้องมี ground truth ที่รู้คำตอบเป๊ะ ๆ (หลุมลึก 8.0 mm กว้าง 12 cm อยู่ที่ x=1.2 m)
ตัวสร้างนี้ให้ทั้ง depth ที่สมจริง (noise ตามสูตร stereo, เงาบัง, ค่าควอนไทซ์ 1 mm)
และเฉลยที่แม่นยำ → ใช้วัด error ของท่อทั้งเส้นได้โดยไม่ต้องมีพื้นที่จริง

โมเดล noise ของ stereo depth:  σ_z = z² · δ_subpix / (f_x · B)
  D455: baseline B ≈ 95 mm, f_x ≈ 447 px ที่ 848×480 (HFOV 87°), δ ≈ 0.08 px
  → σ ≈ 1.9 mm ที่ 1 m, 0.5 mm ที่ 0.5 m — ตรงกับที่วัดได้จริงจากกล้องตัวนี้

และ noise นั้น "ไม่อิสระต่อพิกเซล": D455 หา disparity ด้วย block matching หน้าต่างราว
7 px พิกเซลข้างกันจึงผิดไปทางเดียวกัน (noise_corr_px) ข้อนี้สำคัญกว่าที่คิด — ถ้าจำลอง
เป็น noise อิสระ การเฉลี่ยหลายพิกเซลจะลด error ลงตาม 1/√n ซึ่งทำให้ตัวจำลอง "เก่งเกินจริง"
มาก ๆ ที่ระยะไกล: เซลล์ที่มี 3-4 จุดจะดูแม่นทั้งที่กล้องจริงพ่นหลุมผีลึก 37 mm ตรงนั้น
(วัดเทียบกันมาแล้วบนพรมจริง 12 เฟรม)
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


D455_BASELINE_M = 0.095


@dataclass
class CameraSpec:
    width: int = 848
    height: int = 480
    hfov_deg: float = 87.0             # D455 depth HFOV
    height_m: float = 0.60             # ความสูงกล้องเหนือพื้น
    pitch_deg: float = 45.0            # ก้มลงกี่องศาจากแนวนอน (90 = มองตรงลง)
    subpix_px: float = 0.08            # ความละเอียด disparity → กำหนด noise
    baseline_m: float = D455_BASELINE_M
    depth_unit_m: float = 0.001        # ควอนไทซ์ 1 mm เหมือน RealSense จริง
    dropout_frac: float = 0.02         # พิกเซลหลุดแบบสุ่ม
    noise_corr_px: float = 7.0         # ความกว้างที่ noise สัมพันธ์กัน = หน้าต่าง block matching
                                       # 1 = noise อิสระต่อพิกเซล (ไม่สมจริง แต่ไว้เทียบ)
    zmax: float = 6.0

    @property
    def K(self):
        fx = (self.width / 2.0) / np.tan(np.radians(self.hfov_deg) / 2.0)
        fy = fx
        return (fx, fy, self.width / 2.0 - 0.5, self.height / 2.0 - 0.5)


@dataclass
class Feature:
    """ความผิดปกติหนึ่งจุดบนพื้น (พิกัด floor frame: x=หน้า, y=ซ้าย, หน่วยเมตร)"""
    kind: str                          # 'pothole'|'bump'|'step'|'slope'|'wave'|'rough'
    x: float = 0.0
    y: float = 0.0
    depth_mm: float = 0.0              # ลบ = ลึกลง, บวก = นูนขึ้น (สำหรับ pothole/bump/step)
    radius_m: float = 0.10             # รัศมี (pothole/bump)
    length_m: float = 0.0              # ยาว/กว้างเพิ่มเติมสำหรับ step (0 = ทั้งแนว)
    angle_deg: float = 0.0             # ทิศของ step / slope
    wavelength_m: float = 0.5          # สำหรับ wave
    sharp: bool = False                # ขอบคม (จาน vs หลุมขอบชัน)

    def label(self) -> str:
        return f'{self.kind}@({self.x:.2f},{self.y:.2f}) {self.depth_mm:+.1f}mm'


@dataclass
class FloorScene:
    features: List[Feature] = field(default_factory=list)
    roughness_mm: float = 0.0          # ความหยาบผิวแบบสุ่ม (RMS) เช่น พื้นปูน 0.3-0.8 mm
    roughness_scale_m: float = 0.05
    seed: int = 0

    def height_mm(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ความสูงพื้น (mm) ที่พิกัด (x, y) — เวกเตอร์ไรซ์เต็มรูปแบบ"""
        z = np.zeros_like(x, dtype=np.float64)
        for f in self.features:
            if f.kind in ('pothole', 'bump'):
                r = np.hypot(x - f.x, y - f.y)
                if f.sharp:
                    prof = (r <= f.radius_m).astype(np.float64)
                else:
                    # จานโค้งนุ่ม (cosine bowl) — เหมือนหลุมสึกกร่อนจริงมากกว่าทรงกระบอก
                    prof = np.where(r <= f.radius_m,
                                    0.5 * (1.0 + np.cos(np.pi * np.clip(r / max(f.radius_m, 1e-6), 0, 1))),
                                    0.0)
                z += f.depth_mm * prof
            elif f.kind == 'step':
                th = np.radians(f.angle_deg)
                s = (x - f.x) * np.cos(th) + (y - f.y) * np.sin(th)
                side = (s >= 0).astype(np.float64)
                if f.length_m > 0:                     # ลิ่มเป็นแถบกว้างจำกัด (เช่น รอยต่อพื้น)
                    side = ((s >= 0) & (s <= f.length_m)).astype(np.float64)
                z += f.depth_mm * side
            elif f.kind == 'slope':
                th = np.radians(f.angle_deg)
                s = (x - f.x) * np.cos(th) + (y - f.y) * np.sin(th)
                z += np.tan(np.radians(f.depth_mm)) * s * 1000.0   # depth_mm ใช้เป็น "องศาลาด"
            elif f.kind == 'wave':
                th = np.radians(f.angle_deg)
                s = (x - f.x) * np.cos(th) + (y - f.y) * np.sin(th)
                z += f.depth_mm * np.sin(2 * np.pi * s / max(f.wavelength_m, 1e-6))
        if self.roughness_mm > 0:
            z += _value_noise(x, y, self.roughness_scale_m, self.seed) * self.roughness_mm
        return z


def _value_noise(x, y, scale, seed):
    """ความหยาบผิวแบบสุ่มที่ต่อเนื่อง (value noise + bilinear) — RMS ≈ 1

    ใช้ hash เชิงตัวเลขแทน lookup table เพื่อให้เรียกซ้ำที่พิกัดเดิมได้ค่าเดิมเสมอ
    (สำคัญ: ray casting ต้องเรียก height_mm หลายรอบที่พิกัดใกล้เคียงกัน)
    """
    gx = x / scale
    gy = y / scale
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    fx = gx - x0
    fy = gy - y0

    def h(ix, iy):
        n = (ix * np.int64(374761393) + iy * np.int64(668265263)
             + np.int64(seed) * np.int64(2246822519))
        n = (n ^ (n >> np.int64(13))) * np.int64(1274126177)
        n = n ^ (n >> np.int64(16))
        return (n & np.int64(0xFFFFFF)) / float(0x7FFFFF) - 1.0

    # smoothstep ให้อนุพันธ์ต่อเนื่อง ไม่เกิดสันเหลี่ยมตามเส้นกริด
    sx = fx * fx * (3 - 2 * fx)
    sy = fy * fy * (3 - 2 * fy)
    v00, v10 = h(x0, y0), h(x0 + 1, y0)
    v01, v11 = h(x0, y0 + 1), h(x0 + 1, y0 + 1)
    a = v00 + (v10 - v00) * sx
    b = v01 + (v11 - v01) * sx
    return (a + (b - a) * sy) * 1.6      # ปรับให้ RMS ใกล้ 1


def _R_floor_from_cam(pitch_deg: float) -> np.ndarray:
    """เมทริกซ์หมุน optical(x ขวา, y ลง, z หน้า) → floor(X หน้า, Y ซ้าย, Z ขึ้น)"""
    th = np.radians(pitch_deg)
    x_opt = np.array([0.0, -1.0, 0.0])
    y_opt = np.array([-np.sin(th), 0.0, -np.cos(th)])
    z_opt = np.array([np.cos(th), 0.0, -np.sin(th)])
    return np.stack([x_opt, y_opt, z_opt], axis=1)


def _correlated_noise(shape, corr_px: float, rng) -> np.ndarray:
    """สนาม noise ที่ค่าเฉลี่ย 0 ความแปรปรวนต่อพิกเซล 1 แต่พิกเซลใกล้กันสัมพันธ์กัน

    ทำด้วย box blur แล้ว normalise variance กลับ — ผลคือ "แต่ละพิกเซลแรงเท่าเดิม
    แต่เฉลี่ยกันแล้วไม่ลดลงตาม 1/√n" ซึ่งเป็นพฤติกรรมที่วัดได้จริงจาก stereo
    """
    n = rng.normal(0.0, 1.0, shape).astype(np.float32)
    k = int(round(corr_px))
    if k < 2:
        return n
    k |= 1
    pad = k // 2
    a = np.pad(n, pad, mode='reflect')
    c = np.cumsum(np.cumsum(a, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    H, W = shape
    box = (c[k:k + H, k:k + W] - c[0:H, k:k + W]
           - c[k:k + H, 0:W] + c[0:H, 0:W]) / float(k * k)
    sd = float(box.std())
    return box / sd if sd > 1e-9 else n


def render_depth(scene: FloorScene, cam: CameraSpec, add_noise: bool = True,
                 occlusion: bool = True, rng: Optional[np.random.Generator] = None):
    """คืน (depth_mm uint16, K, info dict)

    วิธี: ray marching หา "จุดตัดแรก" ของรังสีกับ height field
      1. หาแถบความสูง [z_lo, z_hi] ที่ผิวพื้นเป็นไปได้ → ได้ช่วง t ที่แคบมาก (พื้นเบี่ยงแค่ซม.)
      2. เดินทีละก้าวในช่วงนั้น หาช่วงที่ g(t) = z_รังสี − z_ผิว เปลี่ยนเครื่องหมาย
      3. บีบด้วย bisection

    ทำไมไม่ใช้วิธีวนแทนค่า (ตัดกับ z=0 แล้ววนปรับ) ซึ่งเร็วกว่า: มันไม่ลู่เข้าตรงขอบคม
    รังสีจะกระโดดเข้า-ออกขอบแผ่นไปมา ได้เรขาคณิตผิดรอบ ๆ ขอบทุกอัน — แล้วผลตรวจจับที่
    ได้ก็จะ "ถูกต้องเทียบกับฉากที่ผิด" ซึ่งอันตรายกว่าไม่ทดสอบเลย
    การเดินหาจุดตัดแรกยังให้ "เงาบัง" มาฟรี ๆ และถูกต้องตามฟิสิกส์ด้วย
    """
    rng = rng or np.random.default_rng(0)
    fx, fy, cx, cy = cam.K
    H, W = cam.height, cam.width
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    d_opt = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)

    R = _R_floor_from_cam(cam.pitch_deg)
    C = np.array([0.0, 0.0, cam.height_m])
    d_flr = d_opt @ R.T                                  # ทิศรังสีในเฟรมพื้น
    dz = d_flr[..., 2]
    going_down = dz < -1e-6                              # เฉพาะรังสีที่พุ่งลงถึงพื้น
    dzs = np.where(going_down, dz, -1.0)

    # ขอบเขตความสูงของผิวพื้นทั้งฉาก (สุ่มตารางหยาบพอ) → กำหนดช่วง t ที่ต้องเดิน
    t_flat = np.where(going_down, -C[2] / dzs, np.nan)
    with np.errstate(invalid='ignore'):
        probe_x = C[0] + t_flat * d_flr[..., 0]
        probe_y = C[1] + t_flat * d_flr[..., 1]
    fin = np.isfinite(probe_x) & (t_flat > 0) & (t_flat < cam.zmax * 2)
    if fin.any():
        gx = probe_x[fin][::37]
        gy = probe_y[fin][::37]
        hs = scene.height_mm(gx, gy) * 0.001
        z_hi, z_lo = float(hs.max()) + 0.02, float(hs.min()) - 0.02
    else:
        z_hi, z_lo = 0.02, -0.02

    t_a = (z_hi - C[2]) / dzs                            # เริ่มเดินจากด้าน "เหนือผิว" เสมอ
    t_b = (z_lo - C[2]) / dzs
    n_march = 48
    found = np.zeros((H, W), bool)
    lo = np.zeros((H, W)); hi = np.zeros((H, W))
    t_prev = t_a
    for i in range(1, n_march + 1):
        t_i = t_a + (t_b - t_a) * (i / n_march)
        qx = C[0] + t_i * d_flr[..., 0]
        qy = C[1] + t_i * d_flr[..., 1]
        qz = C[2] + t_i * d_flr[..., 2]
        surf = scene.height_mm(qx, qy) * 0.001
        cross = (~found) & going_down & (qz <= surf)
        lo = np.where(cross, t_prev, lo)
        hi = np.where(cross, t_i, hi)
        found |= cross
        t_prev = t_i
    for _ in range(10):                                  # bisection ให้แม่นระดับ <0.1 mm
        mid = 0.5 * (lo + hi)
        qx = C[0] + mid * d_flr[..., 0]
        qy = C[1] + mid * d_flr[..., 1]
        qz = C[2] + mid * d_flr[..., 2]
        above = qz > scene.height_mm(qx, qy) * 0.001
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    t = np.where(found, 0.5 * (lo + hi), np.nan)

    valid = found & np.isfinite(t) & (t > 0)
    px = C[0] + t * d_flr[..., 0]
    py = C[1] + t * d_flr[..., 1]
    pz = C[2] + t * d_flr[..., 2]
    occl = going_down & ~found                           # ยิงลงแต่ไม่เจอผิว = โดนบัง/พ้นฉาก

    depth_m = np.where(valid, t, 0.0)                    # t = พิกัด z ของกล้อง = ค่า depth พอดี
    valid &= (depth_m > 0.1) & (depth_m < cam.zmax)

    if add_noise:
        sigma = depth_m ** 2 * cam.subpix_px / (fx * cam.baseline_m)
        depth_m = depth_m + _correlated_noise(depth_m.shape, cam.noise_corr_px, rng) * sigma
    if cam.dropout_frac > 0:
        valid &= rng.random(depth_m.shape) >= cam.dropout_frac

    depth_mm = np.where(valid, np.round(depth_m / cam.depth_unit_m), 0)
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    info = dict(K=cam.K, valid=valid, x=px, y=py, z=pz, occluded=occl,
                sigma_mm=float(np.nanmedian((depth_m[valid] ** 2 * cam.subpix_px /
                                             (fx * cam.baseline_m)) * 1000.0)) if valid.any() else float('nan'))
    return depth_mm, cam.K, info


def render_color(scene: FloorScene, cam: CameraSpec, info: dict,
                 base_bgr=(150, 148, 143)) -> np.ndarray:
    """ภาพสีคู่กัน (พื้นปูนเทา + เงาตามความชัน) — ไว้ทดสอบ overlay แบบไม่ต้องมีกล้อง"""
    H, W = cam.height, cam.width
    x, y, valid = info['x'], info['y'], info['valid']
    img = np.zeros((H, W, 3), np.uint8)
    eps = 0.01
    zc = np.zeros((H, W)); zx = np.zeros((H, W)); zy = np.zeros((H, W))
    zc[valid] = scene.height_mm(x[valid], y[valid])
    zx[valid] = scene.height_mm(x[valid] + eps, y[valid])
    zy[valid] = scene.height_mm(x[valid], y[valid] + eps)
    # เฉดตามความชัน: หลุม/ขอบจะเห็นเป็นเส้นเงาเหมือนภาพจริง
    shade = 1.0 - np.clip((np.abs(zx - zc) + np.abs(zy - zc)) / (eps * 1000.0) * 0.35, 0, 0.75)
    tex = 1.0 + 0.06 * _value_noise(x, y, 0.02, scene.seed + 7)
    for i, b in enumerate(base_bgr):
        ch = np.clip(b * shade * tex, 0, 255)
        img[..., i] = np.where(valid, ch, 30).astype(np.uint8)
    return img


# ───────────────────── ชุดฉากสำเร็จรูปสำหรับทดสอบ ─────────────────────

def scene_flat(roughness_mm: float = 0.4) -> FloorScene:
    return FloorScene(features=[], roughness_mm=roughness_mm)


def scene_calibration_targets() -> FloorScene:
    """ฉาก "แผ่นชิม" มาตรฐาน: เนินความสูงรู้ค่า 3/5/10/20 mm เรียงเป็นแถว
    ตรงกับสนามทดสอบจริงที่แนะนำให้สร้างด้วยแผ่นโฟม/ฟิวเจอร์บอร์ด → เทียบผลกันได้ตรง ๆ"""
    fs = []
    for i, hmm in enumerate((3.0, 5.0, 10.0, 20.0)):
        fs.append(Feature('bump', x=0.55 + i * 0.28, y=0.18, depth_mm=hmm,
                          radius_m=0.06, sharp=True))
        fs.append(Feature('pothole', x=0.55 + i * 0.28, y=-0.18, depth_mm=-hmm,
                          radius_m=0.06, sharp=True))
    return FloorScene(features=fs, roughness_mm=0.3)


def scene_warehouse() -> FloorScene:
    """ฉากคล้ายพื้นโกดังจริง: ลาดเอียงเล็กน้อย + คลื่นยาว + รอยต่อมีลิ่ม + หลุมสึก"""
    return FloorScene(features=[
        Feature('slope', depth_mm=0.8, angle_deg=0.0),            # ลาด 0.8°
        Feature('wave', depth_mm=2.5, wavelength_m=1.2, angle_deg=0.0),
        Feature('step', x=1.10, depth_mm=3.0, angle_deg=0.0, length_m=0.03),  # ลิ่มรอยต่อ 3 mm
        Feature('pothole', x=0.75, y=0.22, depth_mm=-9.0, radius_m=0.11),
        Feature('pothole', x=1.45, y=-0.25, depth_mm=-16.0, radius_m=0.08, sharp=True),
        Feature('bump', x=1.00, y=-0.05, depth_mm=6.0, radius_m=0.09),
    ], roughness_mm=0.5)


SCENES = {
    'flat': scene_flat,
    'targets': scene_calibration_targets,
    'warehouse': scene_warehouse,
}
