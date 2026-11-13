# -*- coding: utf-8 -*-
"""session — บันทึก/อ่านเซสชันการตรวจพื้น + ระบบ annotate (ROS-free)

รูปแบบเดียวกับที่งาน FLOBOT ใช้ (บันทึก → annotate → เอาไปวัดผล/ฝึกโมเดล):
เซสชันหนึ่ง = โฟลเดอร์เดียวที่มีทุกอย่างครบ เปิดย้อนหลังได้โดยไม่ต้องมีกล้อง

  <out>/<timestamp>_<label>/
      meta.json       สรุปผล + รายการ defect + สเปกหุ่น + พารามิเตอร์ที่ใช้
      dem.npz         DEM ดิบ (z, count, spread) + แผนที่ค่าเบี่ยงเบน  ← เปิดวิเคราะห์ซ้ำได้
      color.png       ภาพกล้องขณะนั้น
      overlay.png     ภาพพร้อม overlay
      topdown.png     แผนที่มองจากบน
      profile.png     ภาพตัดขวาง

การ annotate เก็บแยกที่ <out>/annotations.csv เพื่อไม่แตะไฟล์ผลดิบ
(คนตรวจกลับมาแก้ป้ายทีหลังได้ โดยข้อมูลวัดยังเป็นของเดิมเสมอ)
"""
import csv
import json
import os
import time
from typing import List, Optional

import numpy as np

ANNOT_FIELDS = ['ts', 'session', 'defect_idx', 'verdict', 'true_kind',
                'true_depth_mm', 'note', 'by']


def _stamp() -> str:
    return time.strftime('%Y%m%d_%H%M%S')


def save(out_dir: str, res, spec, risk_grid, color_bgr, K, label: str = '') -> dict:
    """บันทึกเซสชันหนึ่งชุด คืน dict สรุปให้แดชบอร์ดแสดงผล"""
    from floor_inspect import render as rd
    from floor_inspect import risk as rk
    from floor_inspect import fnumbers as fn
    import cv2

    safe = ''.join(ch for ch in label if ch.isalnum() or ch in '-_')[:40]
    name = _stamp() + (f'_{safe}' if safe else '')
    path = os.path.join(out_dir, name)
    os.makedirs(path, exist_ok=True)

    dem = res.dem
    np.savez_compressed(
        os.path.join(path, 'dem.npz'),
        z=dem.z, count=dem.count, spread_mm=dem.spread_mm,
        dev_plane_mm=res.dev_plane_mm, dev_local_mm=res.dev_local_mm,
        steps_mm=res.steps_mm,
        slope_deg=res.slope_deg if res.slope_deg is not None else np.zeros(1),
        noise_mm=res.noise_mm if res.noise_mm is not None else np.zeros(1),
        risk=risk_grid if risk_grid is not None else np.zeros(1),
        cell=dem.cell, x_max=dem.x_max, y_max=dem.y_max,
        plane_n=res.plane.n, plane_d=res.plane.d,
        T_floor_cam=res.T_floor_cam if res.T_floor_cam is not None else np.eye(4))

    if color_bgr is not None:
        cv2.imwrite(os.path.join(path, 'color.png'), color_bgr)
        ov = rd.overlay_on_color(color_bgr, res, risk_grid, 'risk',
                                 defects=res.defects, K=K)
        cv2.imwrite(os.path.join(path, 'overlay.png'), ov)
    cv2.imwrite(os.path.join(path, 'topdown.png'),
                rd.topdown_map(res, 'dev', defects=res.defects, risk_score=risk_grid))
    cv2.imwrite(os.path.join(path, 'topdown_risk.png'),
                rd.topdown_map(res, 'risk', defects=res.defects, risk_score=risk_grid))
    cv2.imwrite(os.path.join(path, 'profile.png'), rd.profile_image(res, 0.0))

    summ = rk.area_summary(risk_grid, dem.cell) if risk_grid is not None else {}
    verdict, msg = rk.overall_verdict(summ, res.defects, spec)
    f = fn.f_numbers(dem)
    _, gap06 = fn.straightedge_gap_mm(dem, 0.6)
    _, gap2 = fn.straightedge_gap_mm(dem, 2.0)

    meta = dict(
        name=name, label=label, ts=time.time(), note=res.note,
        camera=dict(K=list(K) if K is not None else None,
                    height_m=round(res.plane.cam_height_m, 4),
                    plane_rms_mm=round(res.plane.rms_mm, 3),
                    inlier_frac=round(res.plane.inlier_frac, 4)),
        area=dict(surveyed_m2=round(dem.area_m2, 4), coverage=round(dem.coverage, 4),
                  cell_mm=round(dem.cell * 1000, 2),
                  extent_m=[round(v, 3) for v in dem.extent_m()],
                  detectable_mm=round(res.detectable_mm, 3)),
        flatness=dict(**f.to_dict(),
                      FF_class=fn.classify_FF(f.FF),
                      gap_600mm_mm=None if not np.isfinite(gap06) else round(gap06, 2),
                      gap_2000mm_mm=None if not np.isfinite(gap2) else round(gap2, 2)),
        amr=dict(name=spec.name, caster_d_m=spec.caster_d_m,
                 drive_wheel_d_m=spec.drive_wheel_d_m,
                 step_limit_mm=round(spec.step_limit_mm, 2),
                 max_slope_deg=spec.max_slope_deg, max_speed_ms=spec.max_speed_ms),
        risk=dict(verdict=verdict, message=msg, **summ),
        defects=[d.to_dict() for d in res.defects])
    with open(os.path.join(path, 'meta.json'), 'w') as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    return dict(ok=True, name=name, path=path, verdict=verdict,
                n_defects=len(res.defects), area_m2=meta['area']['surveyed_m2'])


def list_sessions(out_dir: str) -> List[dict]:
    """รายการเซสชันที่บันทึกไว้ ใหม่สุดขึ้นก่อน"""
    if not os.path.isdir(out_dir):
        return []
    rows = []
    for name in sorted(os.listdir(out_dir), reverse=True):
        meta_p = os.path.join(out_dir, name, 'meta.json')
        if not os.path.exists(meta_p):
            continue
        try:
            with open(meta_p) as fp:
                m = json.load(fp)
        except Exception:
            continue
        rows.append(dict(name=name, label=m.get('label', ''), ts=m.get('ts', 0),
                         verdict=m.get('risk', {}).get('verdict', '—'),
                         n_defects=len(m.get('defects', [])),
                         area_m2=m.get('area', {}).get('surveyed_m2', 0),
                         FF=m.get('flatness', {}).get('FF')))
    return rows


def load(session_path: str):
    """อ่านเซสชันกลับมาเป็น (meta, arrays) — วิเคราะห์ซ้ำ/ทำรายงานย้อนหลังได้โดยไม่ต้องมีกล้อง"""
    with open(os.path.join(session_path, 'meta.json')) as fp:
        meta = json.load(fp)
    arr = dict(np.load(os.path.join(session_path, 'dem.npz'), allow_pickle=False))
    return meta, arr


def annotate(out_dir: str, payload: dict) -> dict:
    """บันทึกป้ายกำกับจากคนตรวจ (ยืนยัน/ปฏิเสธ/แก้ชนิด) ลง annotations.csv

    ใช้วัด precision/recall ของระบบกับพื้นจริง — ตัวเลขที่ต้องมีถ้าจะเอาไปเขียนรายงาน
    verdict: 'tp' ตรวจถูก | 'fp' ตรวจผิด (ไม่มีจริง) | 'fn' มีจริงแต่ไม่ตรวจเจอ | 'unsure'
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'annotations.csv')
    new = not os.path.exists(path)
    row = {k: payload.get(k, '') for k in ANNOT_FIELDS}
    row['ts'] = time.time()
    if row['verdict'] not in ('tp', 'fp', 'fn', 'unsure'):
        return {'ok': False, 'error': "verdict ต้องเป็น tp / fp / fn / unsure"}
    with open(path, 'a', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=ANNOT_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return {'ok': True, 'path': path}


def annotation_stats(out_dir: str) -> dict:
    """สรุป precision / recall จากป้ายที่คนตรวจกำกับไว้"""
    path = os.path.join(out_dir, 'annotations.csv')
    if not os.path.exists(path):
        return dict(n=0, note='ยังไม่มีการ annotate')
    tp = fp = fn_ = 0
    with open(path) as fpp:
        for r in csv.DictReader(fpp):
            v = (r.get('verdict') or '').strip()
            tp += v == 'tp'
            fp += v == 'fp'
            fn_ += v == 'fn'
    prec = tp / (tp + fp) if (tp + fp) else float('nan')
    rec = tp / (tp + fn_) if (tp + fn_) else float('nan')
    f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) \
        and (prec + rec) > 0 else float('nan')
    return dict(n=tp + fp + fn_, tp=tp, fp=fp, fn=fn_,
                precision=round(prec, 3) if np.isfinite(prec) else None,
                recall=round(rec, 3) if np.isfinite(rec) else None,
                f1=round(f1, 3) if np.isfinite(f1) else None)
