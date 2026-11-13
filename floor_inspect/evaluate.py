# -*- coding: utf-8 -*-
"""evaluate — วัดว่าท่อตรวจจับ "แม่นแค่ไหน" เทียบเฉลยที่รู้เป๊ะ ๆ จาก synth (ROS-free)

นี่คือส่วนที่ทำให้ผลไปใช้อ้างอิงในเอกสารได้จริง: แทนที่จะบอกว่า "ตรวจเจอหลุม" เฉย ๆ
เราตอบได้เป็นตัวเลขว่า — ที่ระยะ 1.0 m หลุมลึก 5 mm ตรวจเจอกี่ %, ความลึกที่วัดได้
คลาดเคลื่อนเฉลี่ยกี่ mm, และมี false positive กี่จุดต่อตารางเมตร

จับคู่ detection กับเฉลยด้วยระยะทางบนพื้น (เกณฑ์ match_tol_m) แล้วคิด
  recall    = เฉลยที่ถูกตรวจเจอ / เฉลยทั้งหมด
  precision = detection ที่ตรงเฉลย / detection ทั้งหมด (ไม่นับชนิด 'missing')
  bias/RMSE ของความลึกที่วัดได้
"""
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np

from floor_inspect import synth
from floor_inspect import surface_core as sc


@dataclass
class MatchResult:
    truth_kind: str
    truth_depth_mm: float
    truth_x: float
    truth_y: float
    range_m: float
    detected: bool
    det_kind: str = ''
    det_depth_mm: float = float('nan')
    det_dist_mm: float = float('nan')
    snr: float = float('nan')
    noise_mm: float = float('nan')

    @property
    def depth_err_mm(self) -> float:
        if not self.detected or not np.isfinite(self.det_depth_mm):
            return float('nan')
        return self.det_depth_mm - self.truth_depth_mm


def _truth_points(scene: synth.FloorScene) -> List[Dict]:
    out = []
    for f in scene.features:
        if f.kind in ('pothole', 'bump'):
            out.append(dict(kind=f.kind, x=f.x, y=f.y, depth_mm=f.depth_mm,
                            radius_m=f.radius_m))
    return out


def evaluate_scene(scene: synth.FloorScene, cam: synth.CameraSpec,
                   p: Optional[sc.SurfaceParams] = None,
                   match_tol_m: float = 0.12, seed: int = 0,
                   n_frames: int = 1):
    """รันท่อกับฉากหนึ่ง แล้วจับคู่ผลกับเฉลย

    n_frames > 1 = โหมดสแกนนิ่ง (สะสมหลายเฟรมเฉลี่ยกัน) — ใช้ดูว่าการสะสมช่วยได้จริงแค่ไหน
    """
    p = p or sc.SurfaceParams()
    rng = np.random.default_rng(seed)

    if n_frames <= 1:
        depth, K, _ = synth.render_depth(scene, cam, rng=rng)
        res = sc.analyse_depth(depth, K, p)
    else:
        res = analyse_accumulated(scene, cam, p, n_frames=n_frames, rng=rng)

    truths = _truth_points(scene)
    if not res.ok or res.dem is None:
        return [MatchResult(t['kind'], t['depth_mm'], t['x'], t['y'],
                            float(np.hypot(t['x'], cam.height_m)), False)
                for t in truths], res, []

    # นับเฉพาะเฉลยที่อยู่ในพื้นที่ที่สำรวจได้จริง — เฉลยนอกกริดไม่ควรถูกนับว่า "พลาด"
    #
    # "ในกริด" ไม่พอ: ท่อจะตัดบริเวณที่วัดไม่ไหวออก (ข้อมูลบาง/เกณฑ์โตเกิน) โดยทำเป็น
    # NaN คาไว้ในกริดเดิม ถ้าใช้แค่กรอบสี่เหลี่ยม เฉลยที่อยู่ในบริเวณที่ระบบ "ประกาศแล้ว
    # ว่าไม่รับรอง" จะถูกนับเป็น miss ทั้งที่ระบบทำถูก — ต้องเช็คว่าเซลล์นั้นยังมีค่าจริง
    d = res.dem
    x_lo = d.x_max - d.shape[0] * d.cell
    y_lo = d.y_max - d.shape[1] * d.cell

    def _surveyed(t) -> bool:
        if not (x_lo + 0.03 <= t['x'] <= d.x_max - 0.03
                and y_lo + 0.03 <= t['y'] <= d.y_max - 0.03):
            return False
        r0, c0 = d.rc_of(t['x'], t['y'])
        r0, c0 = int(r0), int(c0)
        k = max(1, int(round(0.03 / d.cell)))
        sl = (slice(max(0, r0 - k), r0 + k + 1), slice(max(0, c0 - k), c0 + k + 1))
        return bool(d.valid[sl].any())

    in_view = [t for t in truths if _surveyed(t)]

    cands = [f for f in res.defects if f.kind in ('pothole', 'bump', 'step', 'obstacle')]
    used = set()
    matches: List[MatchResult] = []
    for t in in_view:
        best_i, best_d = -1, 1e9
        for i, f in enumerate(cands):
            if i in used:
                continue
            dd = float(np.hypot(f.x - t['x'], f.y - t['y']))
            if dd < best_d:
                best_d, best_i = dd, i
        if best_i >= 0 and best_d <= match_tol_m:
            f = cands[best_i]
            used.add(best_i)
            # 'step' รายงานขนาดแบบไม่มีเครื่องหมาย — ใส่เครื่องหมายกลับจาก mean_mm
            dep = f.peak_mm
            if f.kind == 'step' and np.isfinite(f.mean_mm):
                dep = abs(f.peak_mm) * (1.0 if f.mean_mm >= 0 else -1.0)
            matches.append(MatchResult(t['kind'], t['depth_mm'], t['x'], t['y'],
                                       float(np.hypot(t['x'], cam.height_m)), True,
                                       f.kind, float(dep), best_d * 1000.0,
                                       f.snr, f.noise_mm))
        else:
            matches.append(MatchResult(t['kind'], t['depth_mm'], t['x'], t['y'],
                                       float(np.hypot(t['x'], cam.height_m)), False))
    false_pos = [f for i, f in enumerate(cands) if i not in used]
    return matches, res, false_pos


def analyse_accumulated(scene: synth.FloorScene, cam: synth.CameraSpec,
                        p: sc.SurfaceParams, n_frames: int = 8,
                        rng: Optional[np.random.Generator] = None) -> sc.SurfaceResult:
    """โหมด "จอดนิ่งแล้วสแกน": สะสม DEM หลายเฟรมก่อนวิเคราะห์

    เป็นวิธีมาตรฐานของงานสแกนพื้น — noise แบบสุ่มลดลงตาม √N ทำให้เห็นความผิดปกติ
    ระดับ 2-3 mm ที่เฟรมเดียวมองไม่เห็น
    """
    rng = rng or np.random.default_rng(0)
    acc = None
    plane = None
    K = None
    for _ in range(n_frames):
        depth, K, _ = synth.render_depth(scene, cam, rng=rng)
        xyz, valid = sc.deproject_full(depth, K, p.zmin, p.zmax)
        if valid.sum() < 500:
            continue
        if plane is None:                 # ล็อกระนาบจากเฟรมแรก — กล้องไม่ขยับระหว่างสแกน
            sub = xyz[::p.stride_fit, ::p.stride_fit][valid[::p.stride_fit, ::p.stride_fit]]
            plane = sc.fit_reference_plane(sub, p)
            if not plane.ok:
                return sc.SurfaceResult(ok=False, plane=plane, note='หาระนาบไม่เจอ')
            T = sc.floor_transform(plane)
            bounds = None
        pf = sc.transform_points(xyz, T)
        dem = sc.build_dem(pf[valid], p, bounds=bounds)
        if dem is None:
            continue
        if bounds is None:                # ตรึงขอบกริดจากเฟรมแรก ให้ทุกเฟรมซ้อนกันได้พอดี
            bounds = (dem.x_max - dem.shape[0] * dem.cell, dem.x_max,
                      dem.y_max - dem.shape[1] * dem.cell, dem.y_max)
        acc = sc.accumulate_dem(acc, dem)
    if acc is None:
        return sc.SurfaceResult(ok=False, plane=plane or sc.Plane(np.array([0., -1., 0.]), 0.),
                                note='สะสมข้อมูลไม่ได้')
    dem_avg, _ = sc.finish_accumulation(acc, min_frames=max(2, n_frames // 3))
    dem_avg = sc.crop_to_surveyed(dem_avg, p.solid_frac, p.solid_win_m)
    noise = sc.local_noise_mm(dem_avg, p.noise_win_m)
    dem_s = sc.median_prefilter(dem_avg, p.median_pre)
    dev_plane, dev_local, _ = sc.deviation_maps(dem_s, p, noise)
    steps = sc.step_map(dem_s, p.step_base_m)
    rough, _ = sc.surface_floor_mm(
        sc.local_roughness_mm(dev_local, dem_s.cell, p.rough_win_m),
        sc.dem_range_m(dem_s, T), dem_s.valid)
    defects = sc.merge_overlapping(
        sc.detect_defects(dem_s, dev_local, steps, noise, p, plane.cam_height_m,
                          sc.solid_mask(dem_s, p.solid_win_m, p.solid_frac),
                          rough_mm=rough))
    return sc.SurfaceResult(ok=True, plane=plane, T_floor_cam=T, dem=dem_s,
                            dev_plane_mm=dev_plane, dev_local_mm=dev_local,
                            steps_mm=steps, noise_mm=noise, rough_mm=rough,
                            defects=defects,
                            note=f'สะสม {n_frames} เฟรม')


# ───────────────────── ชุดทดสอบมาตรฐาน: กวาดความลึก × ระยะ ─────────────────────

def depth_sweep_scene(depths_mm=(2, 3, 5, 8, 12, 20), x0=0.55, dx=0.22,
                      radius_m=0.06, sharp=True, roughness_mm=0.3) -> synth.FloorScene:
    """เนิน/หลุมความลึกไล่ระดับ วางเรียงตามระยะ — จำลอง "แผ่นชิมมาตรฐาน" ที่จะสร้างจริง"""
    fs = []
    for i, h in enumerate(depths_mm):
        fs.append(synth.Feature('bump', x=x0 + i * dx, y=0.18, depth_mm=float(h),
                                radius_m=radius_m, sharp=sharp))
        fs.append(synth.Feature('pothole', x=x0 + i * dx, y=-0.18, depth_mm=-float(h),
                                radius_m=radius_m, sharp=sharp))
    return synth.FloorScene(features=fs, roughness_mm=roughness_mm)


def fixed_depth_scene(depth_mm: float, xs, radius_m=0.06, sharp=True,
                      roughness_mm=0.3) -> synth.FloorScene:
    """ความลึกเดียว วางหลายระยะ — ใช้หา "ระยะไกลสุดที่ยังตรวจเจอ" ของความลึกนั้น"""
    fs = []
    for x in xs:
        fs.append(synth.Feature('bump', x=float(x), y=0.18, depth_mm=float(depth_mm),
                                radius_m=radius_m, sharp=sharp))
        fs.append(synth.Feature('pothole', x=float(x), y=-0.18, depth_mm=-float(depth_mm),
                                radius_m=radius_m, sharp=sharp))
    return synth.FloorScene(features=fs, roughness_mm=roughness_mm)


def summarise(matches: List[MatchResult]) -> dict:
    det = [m for m in matches if m.detected]
    errs = np.array([m.depth_err_mm for m in det if np.isfinite(m.depth_err_mm)])
    return dict(n=len(matches), found=len(det),
                recall=len(det) / len(matches) if matches else float('nan'),
                bias_mm=float(errs.mean()) if errs.size else float('nan'),
                rmse_mm=float(np.sqrt((errs ** 2).mean())) if errs.size else float('nan'),
                max_abs_err_mm=float(np.abs(errs).max()) if errs.size else float('nan'))


def run_capability_matrix(heights=(0.4, 0.6, 0.8), pitches=(45.0,),
                          depths_mm=(2, 3, 5, 8, 12, 20), n_frames=1,
                          p: Optional[sc.SurfaceParams] = None, seed=0, verbose=True):
    """ตารางความสามารถ: ความสูงกล้อง × ความลึก defect × ระยะ → เจอ/ไม่เจอ + error

    ผลลัพธ์คือ "ขอบเขตการใช้งานจริง" ของกล้องตัวนี้ ที่เอาไปวางสเปกติดตั้งบน AMR ได้เลย
    """
    p = p or sc.SurfaceParams()
    rows = []
    for h in heights:
        for pitch in pitches:
            cam = synth.CameraSpec(height_m=h, pitch_deg=pitch)
            scene = depth_sweep_scene(depths_mm=depths_mm)
            matches, res, fps = evaluate_scene(scene, cam, p, seed=seed, n_frames=n_frames)
            area = res.dem.area_m2 if (res.ok and res.dem is not None) else float('nan')
            for m in matches:
                rows.append(dict(cam_h=h, pitch=pitch, truth_mm=m.truth_depth_mm,
                                 range_m=m.range_m, detected=m.detected,
                                 err_mm=m.depth_err_mm, snr=m.snr, noise_mm=m.noise_mm,
                                 kind=m.truth_kind, det_kind=m.det_kind))
            if verbose:
                s = summarise(matches)
                print(f'  h={h}m pitch={pitch}° frames={n_frames}: '
                      f'recall={s["recall"]*100:.0f}% bias={s["bias_mm"]:+.2f}mm '
                      f'rmse={s["rmse_mm"]:.2f}mm FP={len(fps)} '
                      f'area={area:.2f}m² detectable={res.detectable_mm:.2f}mm')
    return rows
