# -*- coding: utf-8 -*-
"""render — ทำผลวิเคราะห์ให้ "ดูแล้วเข้าใจทันที" (ROS-free, ใช้ OpenCV ล้วน จึงเร็วพอสตรีมสด)

ภาพที่ผลิตได้ 3 แบบ ตรงกับที่งานสแกนพื้นจริงใช้กัน:
  1. overlay บนภาพกล้อง  — ระบายสีทับพื้นตามระดับความเสี่ยง แบบภาพ slip-risk map
                            (ต่อพิกเซล ไม่ใช่กรอบสี่เหลี่ยม จึงเห็นรูปร่างหลุมจริง ๆ)
  2. deviation heat map   — มองจากบน เขียว=ได้ระดับ แดง=นูน น้ำเงิน=ยุบ ตามแบบรายงาน
                            floor-flatness ของงานสแกนโกดัง
  3. profile / section    — ตัดขวางตามเส้น ดูรูปร่างหลุมเป็นมิลลิเมตร

ข้อความไทยเรนเดอร์ด้วย PIL + ฟอนต์ Loma (cv2.putText วาดไทยไม่ได้ ได้เป็นสี่เหลี่ยม)
"""
import os
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

_FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/tlwg/Loma.ttf',
    '/usr/share/fonts/truetype/tlwg/Garuda.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]
_font_cache = {}


def _font(size: int):
    if not _HAS_PIL:
        return None
    if size in _font_cache:
        return _font_cache[size]
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                f = ImageFont.truetype(path, size)
                _font_cache[size] = f
                return f
            except Exception:
                continue
    _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def draw_texts(img_bgr: np.ndarray, items):
    """วาดข้อความหลายชิ้นในรอบเดียว

    items = [(xy, text, size, color, bg, anchor), ...]  (bg/anchor ใส่ None ได้)

    เหตุที่ต้องมีตัวนี้: การวาดผ่าน PIL ต้องแปลงภาพทั้งเฟรมไป-กลับหนึ่งรอบ
    ที่ 1280x720 คือราว 15 ms ต่อรอบ ถ้าเรียก draw_text ทีละป้าย 6 ป้าย
    ก็เสีย 90 ms ต่อเฟรม — มากกว่าเวลาที่ใช้วิเคราะห์พื้นทั้งภาพเสียอีก
    """
    if not items:
        return img_bgr
    if not _HAS_PIL:
        if _HAS_CV2:
            for xy, text, size, color, _bg, _anchor in items:
                cv2.putText(img_bgr, text, (int(xy[0]), int(xy[1]) + size),
                            cv2.FONT_HERSHEY_SIMPLEX, size / 32.0, color, 1, cv2.LINE_AA)
        return img_bgr
    pil = Image.fromarray(img_bgr[:, :, ::-1])
    d = ImageDraw.Draw(pil, 'RGBA')
    for xy, text, size, color, bg, anchor in items:
        anchor = anchor or 'la'
        f = _font(size)
        if bg is not None:
            try:
                box = d.textbbox(xy, text, font=f, anchor=anchor)
                d.rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2),
                            fill=(bg[2], bg[1], bg[0], 205))
            except Exception:
                pass
        d.text(xy, text, font=f, fill=(color[2], color[1], color[0], 255), anchor=anchor)
    img_bgr[:] = np.asarray(pil)[:, :, ::-1]
    return img_bgr


def draw_text(img_bgr: np.ndarray, xy, text: str, size: int = 16,
              color=(255, 255, 255), bg=None, anchor='la'):
    """วาดข้อความชิ้นเดียว (รองรับไทย) — ถ้าจะวาดหลายชิ้นให้ใช้ draw_texts"""
    return draw_texts(img_bgr, [(xy, text, size, color, bg, anchor)])


# ───────────────────────────── ชุดสี ─────────────────────────────

def _lut_from_stops(stops) -> np.ndarray:
    """สร้าง LUT 256 สี BGR จากจุดสีที่กำหนด [(ตำแหน่ง 0..1, (B,G,R)), ...]"""
    lut = np.zeros((256, 3), np.uint8)
    pos = np.array([s[0] for s in stops], float)
    cols = np.array([s[1] for s in stops], float)
    t = np.linspace(0, 1, 256)
    for ch in range(3):
        lut[:, ch] = np.clip(np.interp(t, pos, cols[:, ch]), 0, 255).astype(np.uint8)
    return lut


# แผนที่ค่าเบี่ยงเบน: น้ำเงิน = ยุบต่ำ, เขียว = ได้ระดับ, แดง = นูนสูง
# (ตรงตามแบบรายงาน deviation heat map ของงานสแกนพื้นโกดัง)
LUT_DEV = _lut_from_stops([
    (0.00, (150, 40, 20)),      # น้ำเงินเข้ม = ยุบมาก
    (0.25, (220, 140, 40)),     # ฟ้า
    (0.45, (150, 200, 120)),    # เขียวอ่อน
    (0.50, (90, 190, 60)),      # เขียว = อยู่ในเกณฑ์
    (0.55, (150, 200, 120)),
    (0.75, (60, 190, 245)),     # เหลือง/ส้ม
    (1.00, (40, 40, 220)),      # แดง = นูนมาก
])

# แผนที่ความเสี่ยง 4 ระดับ (ok/watch/slow/block)
RISK_COLORS = {
    'ok':    (90, 190, 60),
    'watch': (60, 210, 240),
    'slow':  (40, 130, 245),
    'block': (48, 48, 225),
}
RISK_LABEL_TH = {'ok': 'ปกติ', 'watch': 'เฝ้าระวัง', 'slow': 'ต้องชะลอ', 'block': 'ผ่านไม่ได้'}
KIND_LABEL_TH = {'pothole': 'หลุม', 'bump': 'เนิน', 'step': 'ลิ่ม/ขอบ',
                 'obstacle': 'สิ่งกีดขวาง', 'missing': 'ไม่มีข้อมูล'}


def colorize_dev(dev_mm: np.ndarray, vmax: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    """ค่าเบี่ยงเบน (mm) → ภาพสี BGR + mask ที่มีข้อมูล"""
    v = np.isfinite(dev_mm)
    t = np.clip(np.nan_to_num(dev_mm, nan=0.0) / (2.0 * vmax) + 0.5, 0.0, 1.0)
    idx = np.where(v, (t * 255).astype(np.uint8), 0)
    img = LUT_DEV[idx]
    img[~v] = (25, 25, 28)
    return img, v


def colorize_risk(score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """คะแนนความเสี่ยง → ภาพสี 4 ระดับ (ขั้นบันได ไม่ไล่เฉด เพื่อให้อ่านระดับได้ชัด)

    ทำเป็น "หาดัชนีระดับ แล้วเปิดตารางสีทีเดียว" ไม่ใช่ระบายทับกัน 4 รอบ —
    ที่ 1280x720 การระบายทับเต็มเฟรม 4 รอบกินเวลาพอ ๆ กับการวิเคราะห์ทั้งเฟรม
    """
    from floor_inspect.risk import LEVEL_TH
    v = np.isfinite(score)
    s = np.where(v, score, -1.0)
    idx = ((s >= LEVEL_TH['watch']).astype(np.uint8)
           + (s >= LEVEL_TH['slow'])
           + (s >= LEVEL_TH['block']))
    pal = np.array([RISK_COLORS['ok'], RISK_COLORS['watch'],
                    RISK_COLORS['slow'], RISK_COLORS['block'],
                    (25, 25, 28)], np.uint8)
    idx = np.where(v, idx, 4)
    return pal[idx], v


# ───────────────────── overlay บนภาพกล้อง (แบบภาพ slip-risk) ─────────────────────

def project_floor_to_pixel(pts_floor_xy, T_floor_cam: np.ndarray, K) -> Tuple[np.ndarray, np.ndarray]:
    """(x, y, z) ใน floor frame → พิกัดพิกเซล (u, v) — ใช้ปักป้ายชื่อ defect บนภาพ"""
    T_inv = np.linalg.inv(T_floor_cam)
    P = np.asarray(pts_floor_xy, dtype=np.float64).reshape(-1, 3)
    cam = P @ T_inv[:3, :3].T + T_inv[:3, 3]
    fx, fy, cx, cy = K
    z = np.maximum(cam[:, 2], 1e-6)
    u = cam[:, 0] * fx / z + cx
    v = cam[:, 1] * fy / z + cy
    return u, v


def overlay_on_color(color_bgr: np.ndarray, res, grid: np.ndarray, mode: str = 'risk',
                     alpha: float = 0.55, vmax: float = 15.0,
                     defects: Optional[List] = None, K=None,
                     draw_labels: bool = True, min_show: float = 0.15,
                     max_labels: int = 8,
                     tint_safe: float = 0.14) -> np.ndarray:
    """ระบายสีทับ "ทุกพิกเซลที่เป็นพื้น" ตามค่าในกริด — ได้ภาพแบบ slip-risk map

    ทำแบบต่อพิกเซล (ไม่ใช่กรอบสี่เหลี่ยม) เพราะรูปร่างจริงของหลุมคือข้อมูลสำคัญ
    ที่คนดูต้องเห็น — กรอบสี่เหลี่ยมบอกแค่ "แถวนี้มีอะไร" แต่ไม่บอกว่าล้อจะโดนตรงไหน

    grid: ค่าใน DEM (risk score หรือ dev_mm) ; mode: 'risk' | 'dev'
    min_show: ค่าต่ำกว่านี้ไม่ระบายสี (ปล่อยให้เห็นพื้นจริง) — ทำให้ภาพอ่านง่ายขึ้นมาก
    """
    out = color_bgr.copy()
    if res.dem is None or res.pts_floor is None or res.valid is None:
        return out
    dem = res.dem
    H, W = dem.shape
    xf = res.pts_floor[..., 0]
    yf = res.pts_floor[..., 1]
    r = ((dem.x_max - xf) / dem.cell).astype(np.int32)
    c = ((dem.y_max - yf) / dem.cell).astype(np.int32)
    inside = res.valid & (r >= 0) & (r < H) & (c >= 0) & (c < W)
    if not inside.any():
        return out

    # สร้างที่ความละเอียดของ depth ที่วิเคราะห์จริง แล้วค่อยขยายให้เท่าภาพสี
    # (เมื่อ pixel_stride > 1 สองอย่างนี้ขนาดไม่เท่ากัน)
    val = np.full(res.valid.shape, np.nan, np.float32)
    val[inside] = grid[r[inside], c[inside]]
    if val.shape != out.shape[:2]:
        val = cv2.resize(val, (out.shape[1], out.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    has = np.isfinite(val)

    if mode == 'risk':
        layer, _ = colorize_risk(val)
        strength = np.clip(np.abs(val) / 1.0, 0, 1)
    else:
        layer, _ = colorize_dev(val, vmax)
        strength = np.clip(np.abs(val) / max(vmax, 1e-6), 0, 1)
    # ย้อมพื้นที่ "ตรวจแล้วและปลอดภัย" ด้วยสีจาง ๆ — บอกขอบเขตที่วิเคราะห์จริง
    # (ถ้าไม่ย้อม คนดูแยกไม่ออกระหว่าง "ตรงนี้พื้นดี" กับ "ตรงนี้กล้องมองไม่ถึง")
    a = np.where(strength >= min_show, alpha * np.clip(strength, 0, 1), tint_safe)
    a = np.where(has, a, 0.0)
    if _HAS_CV2:
        # ผสมด้วยเลขจำนวนเต็ม 8 บิตผ่าน cv2 (มี SIMD) แทนการแปลงทั้งเฟรมเป็น float32
        # สองก้อน — ที่ 1280x720 ทางเดิมกินราว 60 ms ทางนี้เหลือหลักไม่กี่ ms
        a8 = np.clip(a * 255.0, 0, 255).astype(np.uint8)
        a3 = cv2.merge([a8, a8, a8])
        inv = cv2.bitwise_not(a3)
        out = cv2.add(cv2.multiply(out, inv, scale=1.0 / 255.0),
                      cv2.multiply(layer, a3, scale=1.0 / 255.0))
    else:
        a = a[..., None]
        out = (out.astype(np.float32) * (1 - a)
               + layer.astype(np.float32) * a).astype(np.uint8)

    # ขอบเขตของบริเวณเสี่ยง — ช่วยให้เห็น "ขอบหลุม" ชัดแม้สีจาง
    if _HAS_CV2 and mode == 'risk':
        from floor_inspect.risk import LEVEL_TH
        for lvl, thr in (('block', LEVEL_TH['block']), ('slow', LEVEL_TH['slow'])):
            m = (has & (val >= thr)).astype(np.uint8)
            if m.sum() < 30:
                continue
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, [q for q in cs if cv2.contourArea(q) > 60], -1,
                             RISK_COLORS[lvl], 2, cv2.LINE_AA)

    if draw_labels and defects and K is not None and res.T_floor_cam is not None:
        pts = np.array([[d.x, d.y, 0.0] for d in defects], float)
        if len(pts):
            u, v = project_floor_to_pixel(pts, res.T_floor_cam, K)
            # กากบาททุกจุด แต่ "ป้ายข้อความ" ให้เฉพาะจุดที่สำคัญที่สุด
            #
            # พื้นจริงที่มีของวางเยอะ (สายไฟ อะแดปเตอร์ ขอบเคส) ให้ผลได้ 30-40 จุดต่อเฟรม
            # ซึ่งแต่ละจุดถูกต้องหมด แต่ถ้าเขียนป้ายครบทุกจุด ป้ายจะทับกันจนอ่านอะไรไม่ได้เลย
            # และดูเหมือน "ตรวจมั่ว" ทั้งที่ไม่ได้มั่ว — ภาพต้องชี้ของที่ต้องจัดการก่อน
            on_screen = [(d, int(uu), int(vv)) for d, uu, vv in zip(defects, u, v)
                         if 0 <= uu < out.shape[1] and 0 <= vv < out.shape[0]]
            for d, uu, vv in on_screen:
                col = RISK_COLORS.get(getattr(d, 'risk', 'ok'), (200, 200, 200))
                if _HAS_CV2:
                    cv2.drawMarker(out, (uu, vv), col, cv2.MARKER_CROSS, 14, 2)

            n_max = max(0, int(max_labels))
            ranked = sorted(on_screen,
                            key=lambda t: -float(getattr(t[0], 'risk_score', 0.0) or 0.0))
            labels, placed = [], []
            gap = max(28, int(0.035 * out.shape[1]))     # ระยะขั้นต่ำระหว่างป้าย
            for d, uu, vv in ranked:
                if len(labels) >= n_max:
                    break
                if any((uu - pu) ** 2 + (vv - pv) ** 2 < gap * gap for pu, pv in placed):
                    continue                            # ป้ายจะทับของเดิม — ข้ามไป
                placed.append((uu, vv))
                col = RISK_COLORS.get(getattr(d, 'risk', 'ok'), (200, 200, 200))
                kind = KIND_LABEL_TH.get(d.kind, d.kind)
                txt = kind if not np.isfinite(d.peak_mm) else f'{kind} {abs(d.peak_mm):.0f}mm'
                labels.append(((uu + 9, vv - 9), txt, 15, (255, 255, 255), col, None))
            hidden = len(on_screen) - len(labels)
            if hidden > 0:
                labels.append(((out.shape[1] - 10, 8),
                               f'+ อีก {hidden} จุด (ดูรายการในหน้าเว็บ)', 14,
                               (235, 235, 235), (30, 32, 38), 'ra'))
            draw_texts(out, labels)
    return out


# ───────────────────────── แผนที่มองจากบน ─────────────────────────

def topdown_map(res, mode: str = 'dev', vmax: float = 15.0, px_per_cell: int = 3,
                defects: Optional[List] = None, risk_score: Optional[np.ndarray] = None,
                title: str = '', show_grid: bool = True,
                max_labels: int = 8) -> np.ndarray:
    """แผนที่มองจากบนพร้อมสเกล/แถบสี/ป้ายกำกับ — ภาพหลักสำหรับใส่รายงาน

    หน้าขึ้นบน (แถวบน = ไกลจากกล้อง), กล้องอยู่ขอบล่างกลางภาพ
    """
    dem = res.dem
    if dem is None:
        return np.zeros((240, 320, 3), np.uint8)
    show = res.solid if getattr(res, 'solid', None) is not None else dem.valid
    if mode == 'risk' and risk_score is not None:
        base, _ = colorize_risk(np.where(show, risk_score, np.nan))
        unit = ''
    else:
        grid = res.dev_local_mm if mode == 'dev' else res.dev_plane_mm
        base, _ = colorize_dev(np.where(show, grid, np.nan), vmax)
        unit = 'mm'

    k = max(1, int(px_per_cell))
    img = np.repeat(np.repeat(base, k, axis=0), k, axis=1)
    Hp, Wp = img.shape[:2]

    def rc2px(x, y):
        return (int(round((dem.y_max - y) / dem.cell * k)),
                int(round((dem.x_max - x) / dem.cell * k)))

    if show_grid and _HAS_CV2:
        step = 0.25
        x_lo = dem.x_max - dem.shape[0] * dem.cell
        y_lo = dem.y_max - dem.shape[1] * dem.cell
        grid_labels = []
        for xv in np.arange(np.ceil(x_lo / step) * step, dem.x_max, step):
            _, py = rc2px(xv, 0)
            cv2.line(img, (0, py), (Wp, py), (70, 70, 70), 1, cv2.LINE_AA)
            grid_labels.append(((6, max(2, py - 15)), f'x={xv:.2f}m', 12,
                                (185, 190, 200), (30, 32, 38), None))
        draw_texts(img, grid_labels)
        for yv in np.arange(np.ceil(y_lo / step) * step, dem.y_max, step):
            px, _ = rc2px(0, yv)
            cv2.line(img, (px, 0), (px, Hp), (70, 70, 70), 1, cv2.LINE_AA)

    # ตำแหน่งกล้อง = จุดกำเนิด floor frame
    if _HAS_CV2:
        cpx = rc2px(0.0, 0.0)
        if 0 <= cpx[0] < Wp:
            cv2.drawMarker(img, (cpx[0], min(max(cpx[1], 0), Hp - 1)), (255, 255, 255),
                           cv2.MARKER_TRIANGLE_UP, 16, 2)

    if defects and _HAS_CV2:
        # กรอบทุกจุด แต่ป้ายเฉพาะจุดเสี่ยงสูงสุด — เหตุผลเดียวกับใน overlay_on_color
        boxes = []
        for d in defects:
            r0, c0, r1, c1 = d.bbox
            p0 = (c0 * k, r0 * k)
            p1 = ((c1 + 1) * k, (r1 + 1) * k)
            col = RISK_COLORS.get(getattr(d, 'risk', 'ok'), (220, 220, 220))
            cv2.rectangle(img, p0, p1, col, 2)
            boxes.append((d, p0, col))
        n_max = max(0, int(max_labels))
        dlabels, placed = [], []
        gap = max(24, int(0.05 * Wp))
        for d, p0, col in sorted(
                boxes, key=lambda t: -float(getattr(t[0], 'risk_score', 0.0) or 0.0)):
            if len(dlabels) >= n_max:
                break
            if any((p0[0] - qx) ** 2 + (p0[1] - qy) ** 2 < gap * gap for qx, qy in placed):
                continue
            placed.append(p0)
            kind = KIND_LABEL_TH.get(d.kind, d.kind)
            txt = kind if not np.isfinite(d.peak_mm) else f'{kind} {d.peak_mm:+.1f}mm'
            x = min(max(p0[0], 2), max(2, Wp - 110))       # กันป้ายล้นขอบแผนที่
            dlabels.append(((x, max(0, p0[1] - 18)), txt, 14, (255, 255, 255), col, None))
        hidden = len(boxes) - len(dlabels)
        if hidden > 0:
            dlabels.append(((Wp - 8, Hp - 20), f'+ อีก {hidden} จุด', 13,
                            (235, 235, 235), (30, 32, 38), 'ra'))
        draw_texts(img, dlabels)

    img = _add_colorbar(img, mode, vmax, unit)
    header = title or ('แผนที่ความเสี่ยงต่อหุ่น (มองจากบน)' if mode == 'risk'
                       else 'แผนที่ค่าเบี่ยงเบนจากระนาบพื้น (มองจากบน)')
    bar = np.full((34, img.shape[1], 3), (18, 20, 24), np.uint8)
    sub = f'{dem.shape[1]*dem.cell:.2f} × {dem.shape[0]*dem.cell:.2f} m · เซลล์ {dem.cell*1000:.0f} mm'
    draw_texts(bar, [((10, 8), header, 17, (235, 235, 235), None, None),
                     ((img.shape[1] - 10, 10), sub, 13, (150, 160, 175), None, 'ra')])
    return np.vstack([bar, img])


def _add_colorbar(img: np.ndarray, mode: str, vmax: float, unit: str) -> np.ndarray:
    h = 46
    W = img.shape[1]
    bar = np.full((h, W, 3), (18, 20, 24), np.uint8)
    x0, x1, y0, y1 = 12, W - 12, 8, 24
    if x1 <= x0:
        return np.vstack([img, bar])
    t = np.linspace(0, 1, x1 - x0)
    if mode == 'risk':
        from floor_inspect.risk import LEVEL_TH
        strip = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        scale = 1.4
        for i, tv in enumerate(t * scale):
            lvl = ('block' if tv >= LEVEL_TH['block'] else
                   'slow' if tv >= LEVEL_TH['slow'] else
                   'watch' if tv >= LEVEL_TH['watch'] else 'ok')
            strip[:, i] = RISK_COLORS[lvl]
        bar[y0:y1, x0:x1] = strip
        for lbl, tv in (('ปกติ', 0.15), ('เฝ้าระวัง', 0.48), ('ชะลอ', 0.80), ('ผ่านไม่ได้', 1.2)):
            px = int(np.clip(x0 + (tv / 1.4) * (x1 - x0), x0 + 30, x1 - 30))
            draw_text(bar, (px, y1 + 3), lbl, 12, (200, 205, 215), anchor='ma')
    else:
        strip = LUT_DEV[(t * 255).astype(np.uint8)][None, :, :]
        bar[y0:y1, x0:x1] = np.repeat(strip, y1 - y0, axis=0)
        ticks = []
        for frac, val in ((0.0, -vmax), (0.25, -vmax / 2), (0.5, 0.0),
                          (0.75, vmax / 2), (1.0, vmax)):
            px = int(np.clip(x0 + frac * (x1 - x0), x0 + 22, x1 - 22))
            ticks.append(((px, y1 + 3), f'{val:+.0f}{unit}', 12,
                          (200, 205, 215), None, 'ma'))
        draw_texts(bar, ticks)
    return np.vstack([img, bar])


# ───────────────────────────── ภาพตัดขวาง ─────────────────────────────

def profile_image(res, y_m: float = 0.0, width: int = 900, height: int = 260,
                  vmax_mm: float = 25.0) -> np.ndarray:
    """ภาพตัดขวางตามแนวหน้า-หลัง ที่ตำแหน่ง y ที่กำหนด — เห็นรูปทรงหลุมเป็นมิลลิเมตรจริง"""
    img = np.full((height, width, 3), (18, 20, 24), np.uint8)
    dem = res.dem
    if dem is None or not _HAS_CV2:
        return img
    _, c = dem.rc_of(0.0, y_m)
    c = int(np.clip(c, 0, dem.shape[1] - 1))
    prof = dem.z[:, c] * 1000.0
    x_lo = dem.x_max - dem.shape[0] * dem.cell
    xs = np.linspace(dem.x_max, x_lo, dem.shape[0])

    pad_l, pad_b, pad_t = 58, 30, 26
    plot_w, plot_h = width - pad_l - 12, height - pad_b - pad_t

    def to_px(x, z):
        px = pad_l + int((x - x_lo) / max(dem.x_max - x_lo, 1e-6) * plot_w)
        py = pad_t + int((vmax_mm - z) / (2 * vmax_mm) * plot_h)
        return px, np.clip(py, pad_t, pad_t + plot_h)

    ylabels = []
    for zv in np.arange(-vmax_mm, vmax_mm + 1, vmax_mm / 2):
        _, py = to_px(x_lo, zv)
        cv2.line(img, (pad_l, py), (pad_l + plot_w, py),
                 (60, 62, 68) if abs(zv) > 1e-6 else (110, 115, 125), 1)
        ylabels.append(((pad_l - 6, py - 7), f'{zv:+.0f}', 12, (170, 175, 185), None, 'ra'))
    draw_texts(img, ylabels)

    pts = [to_px(x, z) for x, z in zip(xs, prof) if np.isfinite(z)]
    for a, b in zip(pts[:-1], pts[1:]):
        if abs(a[0] - b[0]) < 12:
            cv2.line(img, a, b, (120, 220, 255), 2, cv2.LINE_AA)
    draw_texts(img, [
        ((10, 6), f'ภาพตัดขวางที่ y = {y_m:+.2f} m  (แกนตั้ง: mm)', 15,
         (230, 230, 235), None, None),
        ((width - 12, height - 20), f'x = {x_lo:.2f} ถึง {dem.x_max:.2f} m', 12,
         (150, 160, 175), None, 'ra')])
    return img
