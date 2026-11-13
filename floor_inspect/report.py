# -*- coding: utf-8 -*-
"""report — สร้างรายงานผลตรวจพื้นแบบเอาไปวางในเอกสารได้เลย (ROS-free)

ออกได้ 2 รูปแบบ:
  markdown_report()  ตารางสรุป + รายการ defect + คำตัดสิน — วางใน Word/Docs ได้ตรง ๆ
  figure_png()       รูปประกอบความละเอียดสูงด้วย matplotlib (แผนที่ + ตัดขวาง + ฮิสโตแกรม)

โครงรายงานอิงตาม deliverable ของงานสแกนพื้นโกดังจริง: ค่ามาตรฐาน (F-number) →
แผนที่ค่าเบี่ยงเบน → รายการจุดที่ต้องซ่อม → คำตัดสินเชิงปฏิบัติการ
"""
import os
from typing import Optional

import numpy as np

from floor_inspect import fnumbers as fn
from floor_inspect import risk as rk

_THAI_FONTS = ['Loma', 'Garuda', 'Tlwg Typo', 'DejaVu Sans']

KIND_TH = {'pothole': 'หลุม/แอ่ง', 'bump': 'เนิน/ปูด', 'step': 'ลิ่ม/ขอบต่างระดับ',
           'obstacle': 'สิ่งกีดขวาง', 'missing': 'ไม่มีข้อมูล'}
RISK_TH = {'ok': 'ปกติ', 'watch': 'เฝ้าระวัง', 'slow': 'ต้องชะลอ', 'block': 'ผ่านไม่ได้'}


def markdown_report(res, spec, risk_grid, title: str = 'รายงานผลตรวจสภาพพื้นสำหรับ AMR',
                    annotations: Optional[dict] = None) -> str:
    dem = res.dem
    summ = rk.area_summary(risk_grid, dem.cell) if risk_grid is not None else {}
    verdict, msg = rk.overall_verdict(summ, res.defects, spec)
    f = fn.f_numbers(dem)
    _, gap06 = fn.straightedge_gap_mm(dem, 0.6)
    _, gap2 = fn.straightedge_gap_mm(dem, 2.0)
    L, W = dem.extent_m()

    def g(v, unit='', nd=2):
        return '—' if v is None or (isinstance(v, float) and not np.isfinite(v)) \
            else f'{v:.{nd}f}{unit}'

    out = [f'# {title}', '']
    out += [f'**คำตัดสิน: {verdict}** — {msg}', '']

    out += ['## 1. เงื่อนไขการวัด', '',
            '| รายการ | ค่า |', '|---|---|',
            f'| กล้อง | Intel RealSense D455 (stereo depth) |',
            f'| ความสูงกล้องเหนือพื้น | {g(dem and res.plane.cam_height_m, " m", 3)} |',
            f'| พื้นที่ที่สำรวจได้ | {g(dem.area_m2, " m²")} ({g(L, " m")} × {g(W, " m")}) |',
            f'| ความละเอียดกริด | {dem.cell*1000:.0f} mm/เซลล์ |',
            f'| ความครอบคลุมข้อมูล | {g(dem.coverage*100, " %", 1)} |',
            f'| ความแบนของ fit ระนาบ (RMS) | {g(res.plane.rms_mm, " mm")} |',
            f'| **ความผิดปกติที่เล็กสุดที่ยืนยันได้** | **{g(res.detectable_mm, " mm")}** |',
            f'| หมายเหตุ | {res.note or "วัดจากเฟรมเดียว"} |', '']
    out += ['> "ความผิดปกติที่เล็กสุดที่ยืนยันได้" คำนวณจาก noise ที่วัดได้จริงในเฟรมนี้ '
            '(4σ) — ความผิดปกติที่เล็กกว่านี้อาจมีอยู่จริงแต่แยกจาก noise ไม่ได้ '
            'ถ้าต้องการละเอียดกว่านี้ ให้จอดนิ่งแล้วใช้โหมดสแกนสะสมหลายเฟรม', '']

    out += ['## 2. ดัชนีความเรียบมาตรฐาน', '',
            '| ดัชนี | ค่า | ความหมาย |', '|---|---|---|',
            f'| FF (Flatness, ASTM E1155) | {g(f.FF, "", 1)} | {fn.classify_FF(f.FF)} |',
            f'| FL (Levelness, ASTM E1155) | {g(f.FL, "", 1)} | '
            f'{"—" if f.FL is None else "ยิ่งมากยิ่งได้ระดับ"} |',
            f'| ช่องใต้ไม้บรรทัด 600 mm | {g(gap06, " mm")} | เกณฑ์หน้างานแบบ TR34/ACI |',
            f'| ช่องใต้ไม้บรรทัด 2000 mm | {g(gap2, " mm")} | เกณฑ์หน้างานแบบ TR34/ACI |', '']
    if f.note:
        out += [f'> ⚠ {f.note}', '']

    out += [f'## 3. ความเสี่ยงต่อหุ่น: {spec.name}', '',
            '| พารามิเตอร์หุ่น | ค่า |', '|---|---|',
            f'| ล้อขับ / ล้อประคอง | Ø{spec.drive_wheel_d_m*1000:.0f} / '
            f'Ø{spec.caster_d_m*1000:.0f} mm |',
            f'| ขั้น/ลิ่มสูงสุดที่ผ่านได้ | {spec.step_limit_mm:.1f} mm |',
            f'| ความชันสูงสุด | {spec.max_slope_deg:.1f}° |',
            f'| ความเร็วปกติ | {spec.max_speed_ms:.2f} m/s |', '']
    if summ:
        out += ['| ระดับ | พื้นที่ (m²) | สัดส่วน |', '|---|---|---|']
        tot = max(summ.get('total_m2', 0.0), 1e-9)
        for k in ('ok', 'watch', 'slow', 'block'):
            a = summ.get(k, 0.0)
            out += [f'| {RISK_TH[k]} | {a:.3f} | {100*a/tot:.1f}% |']
        out += ['']

    real = [d for d in res.defects if d.kind != 'missing']
    miss = [d for d in res.defects if d.kind == 'missing']
    out += [f'## 4. รายการความผิดปกติที่พบ ({len(res.defects)} จุด)', '']
    if real:
        out += ['| # | ชนิด | ตำแหน่ง x,y (m) | ขนาด (mm) | ค่าเบี่ยงเบน | ขอบ | ระยะ | SNR | ระดับ | ความเร็วปลอดภัย |',
                '|---|---|---|---|---|---|---|---|---|---|']
        for i, d in enumerate(sorted(real, key=lambda z: -z.risk_score), 1):
            out += [f'| {i} | {KIND_TH.get(d.kind, d.kind)} | {d.x:.2f}, {d.y:+.2f} | '
                    f'{d.length_m*1000:.0f}×{d.width_m*1000:.0f} | {d.peak_mm:+.1f} mm | '
                    f'{d.edge_step_mm:.1f} mm | {d.range_m:.2f} m | {d.snr:.1f} | '
                    f'**{RISK_TH.get(d.risk, d.risk)}** | {d.safe_speed_ms:.2f} m/s |']
        out += ['']
        out += ['### เหตุผลรายจุด', '']
        for i, d in enumerate(sorted(real, key=lambda z: -z.risk_score), 1):
            out += [f'{i}. **{KIND_TH.get(d.kind, d.kind)}** ที่ ({d.x:.2f}, {d.y:+.2f}) — {d.reason}']
        out += ['']
    else:
        out += ['ไม่พบความผิดปกติที่เกินเกณฑ์ในพื้นที่ที่สำรวจ', '']
    if miss:
        out += [f'### บริเวณที่ไม่มีข้อมูล depth ({len(miss)} จุด)', '',
                'จุดเหล่านี้กล้องอ่านค่าไม่ได้ ซึ่ง**ไม่เท่ากับว่าพื้นเรียบ** — '
                'อาจเป็นหลุมลึกที่แสงลงไม่ถึง ผิวสะท้อน/ดำมัน ท่อระบายน้ำ หรือบริเวณที่ถูกบัง '
                'ควรยืนยันด้วยตาก่อนตัดสิน', '',
                '| # | ตำแหน่ง x,y (m) | ขนาด (mm) | พื้นที่ (m²) |', '|---|---|---|---|']
        for i, d in enumerate(miss, 1):
            out += [f'| {i} | {d.x:.2f}, {d.y:+.2f} | {d.length_m*1000:.0f}×'
                    f'{d.width_m*1000:.0f} | {d.area_m2:.4f} |']
        out += ['']

    if annotations and annotations.get('n'):
        out += ['## 5. ความแม่นยำที่ตรวจสอบโดยคน', '',
                '| ตัวชี้วัด | ค่า |', '|---|---|',
                f'| จำนวนที่ตรวจสอบ | {annotations.get("n")} |',
                f'| ตรวจถูก (TP) | {annotations.get("tp")} |',
                f'| ตรวจผิด (FP) | {annotations.get("fp")} |',
                f'| ตรวจไม่เจอ (FN) | {annotations.get("fn")} |',
                f'| Precision | {annotations.get("precision")} |',
                f'| Recall | {annotations.get("recall")} |',
                f'| F1 | {annotations.get("f1")} |', '']

    out += ['---', '',
            '*รายงานนี้สร้างโดย `floor_inspect` — วิธีวัดอ้างอิง ASTM E1155 (F-number), '
            'แนวทางแผนที่ค่าเบี่ยงเบนจากงานสแกนพื้นโกดังด้วย LiDAR, และวิธีสร้างโปรไฟล์พื้น '
            'ด้วยหุ่นและเซนเซอร์ราคาประหยัดตาม Remote Sens. 2019, 11(22), 2626*']
    return '\n'.join(out)


def figure_png(res, spec, risk_grid, path: str, title: str = '') -> Optional[str]:
    """รูปประกอบรายงานความละเอียดสูง (matplotlib) — แผนที่ + ตัดขวาง + การกระจายค่า"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        return None
    plt.rcParams['font.family'] = _THAI_FONTS
    plt.rcParams['axes.unicode_minus'] = False

    dem = res.dem
    x_lo = dem.x_max - dem.shape[0] * dem.cell
    y_lo = dem.y_max - dem.shape[1] * dem.cell
    # DEM แถว 0 = x_max ⇒ ต้องใช้ origin='upper' คู่กับ extent ที่ top = x_max
    # (ถ้าใช้ origin='lower' แกน x จะกลับด้าน — หลุมไปโผล่คนละที่กับของจริง)
    ext = [dem.y_max, y_lo, x_lo, dem.x_max]        # y กลับด้านให้ซ้ายอยู่ซ้าย
    show = res.solid if res.solid is not None else dem.valid

    def masked(a):
        """แสดงเฉพาะเซลล์ที่สำรวจได้จริง — นอกนั้นปล่อยว่าง ไม่ใช่ระบายสีให้ดูเหมือนมีข้อมูล"""
        return np.where(show, a, np.nan)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.28, wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    dev = res.dev_local_mm
    vmax = max(3.0, float(np.nanpercentile(np.abs(dev), 99.5)) if np.isfinite(dev).any() else 5)
    im = ax.imshow(masked(dev), extent=ext, origin='upper', cmap='RdYlGn_r',
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), aspect='equal')
    ax.set_title('ค่าเบี่ยงเบนเฉพาะที่ (flatness)')
    ax.set_xlabel('y : ขวา (ลบ) — ซ้าย (บวก)  (m)')
    ax.set_ylabel('x ระยะหน้าหุ่น (m)')
    fig.colorbar(im, ax=ax, label='mm', shrink=0.85)
    for d in res.defects:
        if d.kind == 'missing':
            continue
        ax.add_patch(plt.Rectangle((d.y + d.width_m / 2, d.x - d.length_m / 2),
                                   -d.width_m, d.length_m, fill=False,
                                   ec={'ok': 'k', 'watch': '#c8a800', 'slow': '#e07000',
                                       'block': '#d02020'}.get(d.risk, 'k'), lw=1.6))

    ax2 = fig.add_subplot(gs[0, 1])
    if risk_grid is not None:
        im2 = ax2.imshow(masked(risk_grid), extent=ext, origin='upper', cmap='RdYlGn_r',
                         vmin=0, vmax=1.4, aspect='equal')
        fig.colorbar(im2, ax=ax2, label='คะแนนเสี่ยง (1.0 = ลิมิตหุ่น)', shrink=0.85)
    ax2.set_title(f'ความเสี่ยงต่อ {spec.name}')
    ax2.set_xlabel('y (m)')
    ax2.set_ylabel('x (m)')

    ax3 = fig.add_subplot(gs[1, 0])
    mid = dem.shape[1] // 2
    xs = np.linspace(dem.x_max, x_lo, dem.shape[0])
    for off, lab, c in ((0, 'กลาง', '#1f77b4'),
                        (-int(0.2 / dem.cell), 'ซ้าย 0.2 m', '#2ca02c'),
                        (int(0.2 / dem.cell), 'ขวา 0.2 m', '#d62728')):
        col = int(np.clip(mid + off, 0, dem.shape[1] - 1))
        ax3.plot(xs, dem.z[:, col] * 1000.0, lw=1.2, label=lab, color=c)
    ax3.axhline(0, color='gray', lw=0.8)
    ax3.set_xlabel('x ระยะหน้าหุ่น (m)')
    ax3.set_ylabel('ความสูงเทียบระนาบ (mm)')
    ax3.set_title('ภาพตัดขวางตามแนวเดิน')
    ax3.grid(alpha=.3)
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    v = masked(dev)[np.isfinite(masked(dev))]
    if v.size:
        ax4.hist(v, bins=80, color='#4c78a8')
        ax4.axvline(0, color='gray', lw=0.8)
        for s, c, lb in ((-spec.step_limit_mm, '#d62728', 'ลิมิตหุ่น'),
                         (spec.step_limit_mm, '#d62728', None)):
            ax4.axvline(s, color=c, ls='--', lw=1.2, label=lb)
        p95 = float(np.percentile(np.abs(v), 95))
        ax4.set_title(f'การกระจายค่าเบี่ยงเบน — 95% อยู่ใน ±{p95:.1f} mm')
    ax4.set_xlabel('mm')
    ax4.set_ylabel('จำนวนเซลล์')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=.3)

    if title:
        fig.suptitle(title, fontsize=14)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path
