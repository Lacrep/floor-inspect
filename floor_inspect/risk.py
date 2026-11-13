# -*- coding: utf-8 -*-
"""risk — แปลง "ความไม่เรียบเป็นมิลลิเมตร" ให้เป็น "หุ่นตัวนี้วิ่งผ่านได้ไหม" (ROS-free)

จุดที่คนมักพลาด: ตัวเลข mm อย่างเดียวตัดสินไม่ได้ ต้องดู "ขนาดด้วย"
  • ร่องกว้าง 5 mm ลึก 50 mm  → ล้อ 150 mm ข้ามฉิว ๆ แทบไม่รู้สึก
  • แอ่งกว้าง 300 mm ลึก 15 mm → ล้อจมลงไปเต็ม 15 mm สะเทือนทั้งคัน
ทั้งสองกรณี "ลึก" ต่างกันลิบ แต่ระบบที่ดูแค่ค่าความลึกจะให้คะแนนสลับกัน

จึงคิดจากเรขาคณิตของล้อจริง: ล้อรัศมี R ข้ามร่องกว้าง w จะจมลงได้มากสุด
    δ = R − √(R² − (w/2)²)      เมื่อ w < 2R  (ถ้า w ≥ 2R คือตกลงไปทั้งล้อ)
ความจมจริง = min(ความลึกร่อง, δ)

ส่วนเนิน/ลิ่ม ใช้เกณฑ์ "ความสูงเทียบเส้นผ่านศูนย์กลางล้อ" ซึ่งเป็นเกณฑ์ที่ผู้ผลิต AMR
ใช้เขียนสเปกกัน และล้อ caster เล็กคือจุดอ่อนเสมอ (iscano: ลิ่ม 3 mm ทำ caster แตกได้)
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

LEVELS = ('ok', 'watch', 'slow', 'block')
LEVEL_TH = {'ok': 0.0, 'watch': 0.35, 'slow': 0.65, 'block': 1.0}


@dataclass
class AMRSpec:
    """สเปกหุ่นที่ใช้ตัดสิน — ค่า default เป็น AMR ขนบรรทุกขนาดกลางทั่วไป"""
    name: str = 'AMR ทั่วไป'
    drive_wheel_d_m: float = 0.150     # ล้อขับ
    caster_d_m: float = 0.075          # ล้อประคอง (ตัวจำกัดจริง — เล็กกว่าเสมอ)
    ground_clearance_m: float = 0.030
    track_width_m: float = 0.450
    max_step_ratio: float = 0.10       # ไต่ขั้นได้ ≤ 10% ของ Ø ล้อ caster อย่างราบรื่น
    max_slope_deg: float = 5.0
    max_speed_ms: float = 1.2
    accel_limit_ms2: float = 2.0       # ความสั่นแนวดิ่งสูงสุดที่ยอมรับได้ (กันของบนหุ่นล้ม)
    payload_sensitive: bool = True

    @property
    def step_limit_mm(self) -> float:
        """ขั้น/ลิ่มสูงสุดที่ผ่านได้ราบรื่น — คิดจากล้อเล็กสุด"""
        return self.caster_d_m * self.max_step_ratio * 1000.0

    @property
    def drop_limit_mm(self) -> float:
        """ความจมของล้อที่ยอมรับได้ — ใช้เกณฑ์เดียวกับขั้น (ขาลงกระแทกพอ ๆ กับขาขึ้น)"""
        return self.step_limit_mm


PRESETS = {
    'amr_generic': AMRSpec(),
    'amr_heavy': AMRSpec(name='AMR ขนของหนัก', drive_wheel_d_m=0.20, caster_d_m=0.10,
                         ground_clearance_m=0.04, track_width_m=0.6, max_slope_deg=4.0,
                         max_speed_ms=1.0, accel_limit_ms2=1.5),
    'amr_small': AMRSpec(name='AMR เล็ก/ส่งของ', drive_wheel_d_m=0.10, caster_d_m=0.05,
                         ground_clearance_m=0.02, track_width_m=0.35, max_slope_deg=6.0,
                         max_speed_ms=1.5, accel_limit_ms2=2.5),
    'agv_tugger': AMRSpec(name='AGV ลากจูง', drive_wheel_d_m=0.25, caster_d_m=0.125,
                          ground_clearance_m=0.05, track_width_m=0.8, max_slope_deg=3.0,
                          max_speed_ms=1.8, accel_limit_ms2=1.2),
}


def wheel_drop_mm(gap_width_m: float, depth_mm: float, wheel_d_m: float) -> float:
    """ล้อจะจมลงไปในร่อง/หลุมนี้จริง ๆ กี่ mm"""
    R = wheel_d_m / 2.0
    w = max(gap_width_m, 0.0)
    if w >= 2.0 * R:
        return abs(depth_mm)                      # กว้างกว่าล้อ → จมสุดความลึก
    delta_mm = (R - np.sqrt(max(R * R - (w / 2.0) ** 2, 0.0))) * 1000.0
    return float(min(abs(depth_mm), delta_mm))


def jolt_ms2(rise_mm: float, length_m: float, speed_ms: float) -> float:
    """ประมาณความเร่งแนวดิ่งที่ตัวหุ่นได้รับตอนข้ามความผิดปกตินี้ (m/s², ประมาณอันดับแรก)

    มองการยกตัวของล้อเป็นครึ่งคลื่นไซน์ยาว length_m ผ่านด้วยความเร็ว speed_ms
    → คาบ T = L/v , แอมพลิจูด A = rise/2 , a_peak = (2π/T)²·A = 2π²·rise·v²/L²
    ใช้เทียบเชิงเปรียบเทียบ/ตั้งลิมิตความเร็ว ไม่ใช่ค่าที่วัดได้จากไดนามิกจริงของตัวรถ
    """
    L = max(length_m, 1e-3)
    return float(2.0 * np.pi ** 2 * (abs(rise_mm) * 0.001) * speed_ms ** 2 / (L * L))


def safe_speed_ms(rise_mm: float, length_m: float, spec: AMRSpec) -> float:
    """ความเร็วสูงสุดที่ข้ามจุดนี้แล้วความสั่นยังไม่เกินลิมิต"""
    if abs(rise_mm) < 1e-6:
        return spec.max_speed_ms
    L = max(length_m, 1e-3)
    v = L * np.sqrt(spec.accel_limit_ms2 / (2.0 * np.pi ** 2 * abs(rise_mm) * 0.001))
    return float(min(spec.max_speed_ms, max(0.05, v)))


def _level_from_score(s: float) -> str:
    if s >= LEVEL_TH['block']:
        return 'block'
    if s >= LEVEL_TH['slow']:
        return 'slow'
    if s >= LEVEL_TH['watch']:
        return 'watch'
    return 'ok'


def assess_defect(d, spec: AMRSpec) -> Tuple[str, float, str, float]:
    """คืน (ระดับความเสี่ยง, คะแนน 0..~2, เหตุผลไทย, ความเร็วปลอดภัย m/s)

    คะแนน 1.0 = ถึงลิมิตของหุ่นพอดี (จุดตัดระหว่าง 'slow' กับ 'block')
    """
    if d.kind == 'missing':
        # ไม่มีข้อมูล = ไม่รู้ว่าข้างล่างเป็นอะไร ต้องถือว่าน่าสงสัยไว้ก่อน
        big = d.area_m2 >= 0.02
        return ('slow' if big else 'watch',
                0.7 if big else 0.4,
                'ไม่มีข้อมูล depth บริเวณนี้ — อาจเป็นหลุมลึก ท่อระบายน้ำ หรือผิวสะท้อน '
                'ต้องยืนยันด้วยตาก่อนปล่อยให้หุ่นวิ่งผ่าน',
                spec.max_speed_ms * 0.5)

    if d.kind == 'obstacle':
        return ('block', 1.5,
                f'สิ่งกีดขวางสูง {abs(d.peak_mm):.0f} mm เกินระยะพ้นพื้น '
                f'{spec.ground_clearance_m*1000:.0f} mm — ต้องหลบ ไม่ใช่ชะลอ',
                0.0)

    peak = abs(d.peak_mm) if np.isfinite(d.peak_mm) else 0.0
    width = d.min_extent_m
    caster = spec.caster_d_m

    if d.kind == 'pothole':
        drop = wheel_drop_mm(width, peak, caster)
        score = drop / max(spec.drop_limit_mm, 1e-6)
        bridged = drop < peak - 0.5
        why = (f'หลุมลึก {peak:.1f} mm กว้าง {width*1000:.0f} mm → ล้อ caster Ø'
               f'{caster*1000:.0f} mm จมจริง {drop:.1f} mm '
               f'(ลิมิต {spec.drop_limit_mm:.1f} mm)')
        if bridged:
            why += ' — ล้อคร่อมปากหลุมได้บางส่วน จึงไม่จมเต็มความลึก'
        rise, span = drop, max(width, 0.02)
    elif d.kind == 'step':
        rise = peak
        score = rise / max(spec.step_limit_mm, 1e-6)
        why = (f'ลิ่ม/ขอบต่างระดับ {rise:.1f} mm (ลิมิต {spec.step_limit_mm:.1f} mm) — '
               'ขอบคมกระแทกล้อโดยตรง อันตรายกว่าแอ่งลาดที่ลึกเท่ากัน')
        span = 0.05                                # ลิ่มขึ้นเต็มความสูงภายในระยะสั้นมาก
    else:                                          # bump
        rise = peak
        eff = min(rise, wheel_drop_mm(width, rise, caster)) if width < caster else rise
        score = eff / max(spec.step_limit_mm, 1e-6)
        why = (f'เนินสูง {rise:.1f} mm กว้าง {width*1000:.0f} mm → ล้อยกจริง {eff:.1f} mm '
               f'(ลิมิต {spec.step_limit_mm:.1f} mm)')
        rise, span = eff, max(width, 0.02)

    # ขอบคมบวกโทษเพิ่ม: ความลึกเท่ากันแต่ขอบชันกระแทกแรงกว่ามาก
    if d.edge_step_mm > spec.step_limit_mm and d.kind != 'step':
        score *= 1.25
        why += f' · ขอบชัน {d.edge_step_mm:.1f} mm เพิ่มแรงกระแทก'

    # ค่าที่ SNR ต่ำ = ยังไม่ยืนยัน ลดน้ำหนักลงแต่ไม่ทิ้ง
    if 0 < d.snr < 5.0:
        score *= 0.8
        why += f' · ความเชื่อมั่นต่ำ (SNR {d.snr:.1f}) ควรวัดซ้ำ'

    v_safe = safe_speed_ms(rise, span, spec)
    return _level_from_score(score), float(score), why, v_safe


def assess_all(defects: List, spec: AMRSpec) -> List:
    """เติมค่า risk/risk_score/reason ให้ทุก defect (แก้ในที่) แล้วคืนลิสต์เดิม"""
    for d in defects:
        lvl, score, why, v = assess_defect(d, spec)
        d.risk, d.risk_score, d.reason = lvl, score, why
        d.safe_speed_ms = v                        # เพิ่มแบบไดนามิก — ใช้ในรายงาน/แดชบอร์ด
    return defects


# ───────────────────── แผนที่ความเสี่ยงต่อเซลล์ (สำหรับ heatmap) ─────────────────────

def risk_grid(dem, dev_local_mm: np.ndarray, steps_mm: Optional[np.ndarray],
              slope_deg: Optional[np.ndarray], spec: AMRSpec,
              noise_mm: Optional[np.ndarray] = None, k_noise: float = 2.0,
              rough_mm: Optional[np.ndarray] = None, k_rough: float = 1.5) -> np.ndarray:
    """คืนกริดคะแนนความเสี่ยงต่อเซลล์ (float, NaN = ไม่มีข้อมูล)

    คะแนนคิดจาก 3 ทาง แล้วเอาที่แย่สุด: ความจม/ยกของล้อ, ลิ่มขอบคม, ความชัน

    ⚠ ต้องหัก "พื้น noise" ออกก่อนเสมอ: ถ้าเอาค่าดิบมาหารลิมิตตรง ๆ พื้นเรียบสนิทที่มี
    noise 1 mm จะได้คะแนน 1/7.5 = 0.13 ทุกเซลล์ และค่า step ที่เป็น noise ล้วน ๆ จะดัน
    หลายเซลล์ข้ามเกณฑ์ 'watch' — แผนที่เลยขึ้นสีเตือนทั้งผืนทั้งที่พื้นไม่มีอะไรเลย
    (วัดจริงกับ synth: ok_pct ตกจาก ~99% เหลือ 48% เพราะเหตุนี้ล้วน ๆ)
    """
    v = dem.valid
    nz = np.where(np.isfinite(noise_mm), noise_mm, 0.0) if noise_mm is not None \
        else np.zeros(dem.shape, np.float32)
    floor_mm = k_noise * nz
    if rough_mm is not None:
        # เนื้อผิวของพื้นเอง (ขนพรม รอยต่อกระเบื้อง) ก็ต้องหักเหมือน noise —
        # ไม่งั้นพรมทั้งผืนขึ้นสีเตือนทั้งที่ล้อกลิ้งข้ามไปไม่รู้สึกอะไร
        rg = np.where(np.isfinite(rough_mm), rough_mm, 0.0)
        floor_mm = np.maximum(floor_mm, k_rough * rg)
    score = np.zeros(dem.shape, np.float32)

    dev = np.where(v & np.isfinite(dev_local_mm), np.abs(dev_local_mm), 0.0)
    # ต่อเซลล์เราไม่รู้ "ความกว้าง" ของความผิดปกติ จึงใช้ค่าเบี่ยงเบนตรง ๆ เทียบลิมิต
    # (การคิดความกว้างจริงทำในระดับ defect ที่ clustering แล้ว — แม่นกว่า)
    score = np.maximum(score, np.maximum(dev - floor_mm, 0.0) / max(spec.drop_limit_mm, 1e-6))

    if steps_mm is not None:
        st = np.where(v & np.isfinite(steps_mm), steps_mm, 0.0)
        st = np.maximum(st - floor_mm * np.sqrt(2.0), 0.0)     # ผลต่างสองฝั่ง → √2·σ
        score = np.maximum(score, st / max(spec.step_limit_mm, 1e-6))

    if slope_deg is not None:
        sl = np.where(v & np.isfinite(slope_deg), slope_deg, 0.0)
        # ความชันเก๊ที่ noise สร้างขึ้นบนฐานวัด 0.20 m
        sl_floor = np.degrees(np.arctan(floor_mm * 0.001 / 0.20))
        sl = np.maximum(sl - sl_floor, 0.0)
        score = np.maximum(score, sl / max(spec.max_slope_deg, 1e-6))

    out = score.astype(np.float32)
    out[~v] = np.nan
    return out


def area_summary(risk_score: np.ndarray, cell_m: float) -> dict:
    """สรุปสัดส่วนพื้นที่ตามระดับความเสี่ยง — ตัวเลขหลักที่ใส่ในรายงาน"""
    v = np.isfinite(risk_score)
    total = float(v.sum()) * cell_m * cell_m
    if total <= 0:
        return dict(total_m2=0.0, ok=0.0, watch=0.0, slow=0.0, block=0.0,
                    ok_pct=float('nan'), worst=float('nan'))
    s = risk_score[v]
    bins = dict(
        ok=float((s < LEVEL_TH['watch']).sum()),
        watch=float(((s >= LEVEL_TH['watch']) & (s < LEVEL_TH['slow'])).sum()),
        slow=float(((s >= LEVEL_TH['slow']) & (s < LEVEL_TH['block'])).sum()),
        block=float((s >= LEVEL_TH['block']).sum()))
    a = cell_m * cell_m
    out = {k: round(n * a, 4) for k, n in bins.items()}
    out['total_m2'] = round(total, 4)
    out['ok_pct'] = round(100.0 * bins['ok'] * a / total, 1)
    out['worst'] = round(float(s.max()), 3)
    return out


def overall_verdict(summary: dict, defects: List, spec: AMRSpec) -> Tuple[str, str]:
    """คำตัดสินรวมของพื้นผืนนี้สำหรับหุ่นตัวนี้ + คำแนะนำที่ทำได้จริง"""
    blocks = [d for d in defects if getattr(d, 'risk', 'ok') == 'block']
    slows = [d for d in defects if getattr(d, 'risk', 'ok') == 'slow']
    if blocks:
        return ('BLOCK', f'พบจุดที่ผ่านไม่ได้ {len(blocks)} จุด — ต้องซ่อมพื้นหรือกำหนดเส้นทางเลี่ยง '
                         'ก่อนปล่อยหุ่นวิ่งบริเวณนี้')
    if slows:
        vmin = min((getattr(d, 'safe_speed_ms', spec.max_speed_ms) for d in slows),
                   default=spec.max_speed_ms)
        return ('SLOW', f'ผ่านได้แต่ต้องชะลอ — พบ {len(slows)} จุดที่ควรลดความเร็วเหลือ '
                        f'~{vmin:.2f} m/s (จากปกติ {spec.max_speed_ms:.2f} m/s)')
    if summary.get('ok_pct', 100) < 95:
        return ('WATCH', 'ผ่านได้ปกติ แต่ผิวมีความไม่เรียบกระจายอยู่ — ควรเฝ้าดูการสึกของล้อ '
                         'และวัดซ้ำเป็นระยะ')
    return ('OK', 'พื้นบริเวณนี้เรียบพอสำหรับหุ่นตัวนี้ วิ่งความเร็วปกติได้')
