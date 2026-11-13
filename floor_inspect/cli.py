# -*- coding: utf-8 -*-
"""cli — รันท่อทั้งเส้นแบบ "ไม่ต้องมีกล้องและไม่ต้องมี ROS"

ใช้ได้ 3 แบบ:
  python3 -m floor_inspect.cli demo   [--scene warehouse] [--out DIR]
      รันกับสนามทดสอบจำลอง → ได้ภาพ overlay/แผนที่/ตัดขวาง + รายงาน .md ครบชุด

  python3 -m floor_inspect.cli bench  [--frames 8]
      กวดความสามารถ: ความสูงกล้อง × ความลึกจริง → ตารางว่าตรวจเจอไหม คลาดเคลื่อนเท่าไร
      นี่คือตารางที่เอาไปใส่เอกสารเป็น "ขอบเขตการใช้งานของกล้องตัวนี้"

  python3 -m floor_inspect.cli replay <session_dir>
      เปิดเซสชันที่บันทึกไว้จากหน้างานจริง มาทำรายงาน/ปรับเกณฑ์ใหม่ย้อนหลัง
"""
import argparse
import os
import sys

import numpy as np


def _write(path, img):
    import cv2
    cv2.imwrite(path, img)
    return path


def cmd_demo(args):
    from floor_inspect import synth, surface_core as sc, risk as rk, render as rd, report as rp
    scene = synth.SCENES[args.scene]()
    cam = synth.CameraSpec(height_m=args.height, pitch_deg=args.pitch)
    print(f'สร้างฉากจำลอง "{args.scene}" — กล้องสูง {args.height} m ก้ม {args.pitch}°')
    depth, K, info = synth.render_depth(scene, cam)
    color = synth.render_color(scene, cam, info)

    p = sc.SurfaceParams(cell=args.cell_mm / 1000.0)
    res = sc.analyse_depth(depth, K, p)
    if not res.ok:
        print('วิเคราะห์ไม่สำเร็จ:', res.note)
        return 1
    spec = rk.PRESETS[args.amr]
    rk.assess_all(res.defects, spec)
    grid = rk.risk_grid(res.dem, res.dev_local_mm, res.steps_mm, res.slope_deg,
                        spec, res.noise_mm, rough_mm=res.rough_mm)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    _write(os.path.join(out, 'color.png'), color)
    _write(os.path.join(out, 'overlay_risk.png'),
           rd.overlay_on_color(color, res, grid, 'risk', defects=res.defects, K=K))
    _write(os.path.join(out, 'overlay_dev.png'),
           rd.overlay_on_color(color, res, res.dev_local_mm, 'dev', defects=res.defects, K=K))
    _write(os.path.join(out, 'topdown_dev.png'),
           rd.topdown_map(res, 'dev', defects=res.defects, risk_score=grid))
    _write(os.path.join(out, 'topdown_risk.png'),
           rd.topdown_map(res, 'risk', defects=res.defects, risk_score=grid))
    _write(os.path.join(out, 'profile.png'), rd.profile_image(res, 0.0))
    rp.figure_png(res, spec, grid, os.path.join(out, 'report_figure.png'),
                  title=f'ฉากจำลอง: {args.scene}')

    md = rp.markdown_report(res, spec, grid,
                            title=f'รายงานผลตรวจพื้น (ฉากจำลอง: {args.scene})')
    md_path = os.path.join(out, 'report.md')
    with open(md_path, 'w') as f:
        f.write(md)

    print(f'\nพบความผิดปกติ {len(res.defects)} จุด · '
          f'พื้นที่ {res.dem.area_m2:.2f} m² · '
          f'เล็กสุดที่ยืนยันได้ {res.detectable_mm:.2f} mm')
    print('\nเทียบกับเฉลยของฉาก:')
    for ft in scene.features:
        if ft.kind in ('pothole', 'bump', 'step'):
            near = min((d for d in res.defects if d.kind != 'missing'),
                       key=lambda d: np.hypot(d.x - ft.x, d.y - ft.y), default=None)
            if near is not None and np.hypot(near.x - ft.x, near.y - ft.y) < 0.12:
                print(f'   {ft.kind:8s} จริง {ft.depth_mm:+6.1f} mm → '
                      f'วัดได้ {near.peak_mm:+6.1f} mm ({near.kind}, {near.risk})')
            else:
                print(f'   {ft.kind:8s} จริง {ft.depth_mm:+6.1f} mm → ไม่พบ '
                      '(ต่ำกว่าเกณฑ์ หรือนอกพื้นที่ที่มองเห็น)')
    print(f'\nไฟล์ผลลัพธ์อยู่ที่: {out}')
    print(f'   รายงาน: {md_path}')
    return 0


def cmd_bench(args):
    from floor_inspect import evaluate as ev, surface_core as sc
    print('กวาดความสามารถ (ยิ่งกล้องสูง ยิ่งเห็นพื้นที่กว้าง แต่ยิ่งเห็นรายละเอียดหยาบ)\n')
    heights = tuple(float(v) for v in args.heights.split(','))
    print(f'{"สูง(m)":>7} {"เฟรม":>5} {"recall":>8} {"bias(mm)":>10} {"rmse(mm)":>9} '
          f'{"FP":>4} {"พื้นที่(m²)":>11} {"เล็กสุด(mm)":>12}')
    print('-' * 74)
    rows = []
    for h in heights:
        for nf in ([1, args.frames] if args.frames > 1 else [1]):
            from floor_inspect import synth
            cam = synth.CameraSpec(height_m=h, pitch_deg=args.pitch)
            scene = ev.depth_sweep_scene()
            m, res, fps = ev.evaluate_scene(scene, cam, sc.SurfaceParams(), n_frames=nf)
            s = ev.summarise(m)
            area = res.dem.area_m2 if (res.ok and res.dem is not None) else float('nan')
            print(f'{h:7.2f} {nf:5d} {s["recall"]*100:7.0f}% {s["bias_mm"]:+10.2f} '
                  f'{s["rmse_mm"]:9.2f} {len(fps):4d} {area:11.2f} {res.detectable_mm:12.2f}')
            rows.append((h, nf, s, len(fps), area, res.detectable_mm))
    print('\nรายละเอียดตามความลึกจริง (กล้องสูง %.2f m, %d เฟรม):' % (heights[0], args.frames))
    from floor_inspect import synth
    cam = synth.CameraSpec(height_m=heights[0], pitch_deg=args.pitch)
    m, res, _ = ev.evaluate_scene(ev.depth_sweep_scene(), cam, sc.SurfaceParams(),
                                  n_frames=args.frames)
    print(f'  {"ลึกจริง":>9} {"ระยะ(m)":>9} {"ตรวจเจอ":>9} {"วัดได้":>10} {"error":>9} {"SNR":>6}')
    for r in sorted(m, key=lambda z: (abs(z.truth_depth_mm), z.truth_kind)):
        got = 'เจอ' if r.detected else 'ไม่เจอ'
        val = f'{r.det_depth_mm:+.1f}' if r.detected else '—'
        err = f'{r.depth_err_mm:+.2f}' if r.detected else '—'
        snr = f'{r.snr:.1f}' if r.detected else '—'
        print(f'  {r.truth_depth_mm:+9.1f} {r.range_m:9.2f} {got:>9} {val:>10} {err:>9} {snr:>6}')
    return 0


def cmd_replay(args):
    from floor_inspect import session as ses, surface_core as sc, risk as rk, report as rp
    meta, arr = ses.load(args.session)
    cell = float(arr['cell'])
    dem = sc.DEM(z=arr['z'], count=arr['count'], spread_mm=arr['spread_mm'],
                 cell=cell, x_max=float(arr['x_max']), y_max=float(arr['y_max']))
    p = sc.SurfaceParams(cell=cell,
                         pothole_mm=args.pothole_mm or 4.0,
                         bump_mm=args.pothole_mm or 4.0)
    noise = sc.local_noise_mm(dem, p.noise_win_m)
    dem_s = sc.median_prefilter(dem, p.median_pre)
    dev_plane, dev_local, _ = sc.deviation_maps(dem_s, p, noise)
    steps = sc.step_map(dem_s, p.step_base_m)
    slope = sc.slope_map(dem_s)
    solid = sc.solid_mask(dem_s, p.solid_win_m, p.solid_frac)
    rough, _ = sc.surface_floor_mm(
        sc.local_roughness_mm(dev_local, dem_s.cell, p.rough_win_m),
        sc.dem_range_m(dem_s, arr['T_floor_cam']), dem_s.valid)
    plane = sc.Plane(n=arr['plane_n'], d=float(arr['plane_d']), ok=True,
                     rms_mm=meta['camera'].get('plane_rms_mm', float('nan')))
    defects = sc.merge_overlapping(sc.detect_defects(
        dem_s, dev_local, steps, noise, p, plane.cam_height_m, solid, rough_mm=rough))
    res = sc.SurfaceResult(ok=True, plane=plane, T_floor_cam=arr['T_floor_cam'], dem=dem_s,
                           dev_plane_mm=dev_plane, dev_local_mm=dev_local, steps_mm=steps,
                           slope_deg=slope, noise_mm=noise, rough_mm=rough, defects=defects,
                           note=f"เปิดซ้ำจาก {meta['name']}")
    spec = rk.PRESETS[args.amr]
    rk.assess_all(res.defects, spec)
    grid = rk.risk_grid(dem_s, dev_local, steps, slope, spec, noise, rough_mm=rough)
    md = rp.markdown_report(res, spec, grid,
                            title=f'รายงานผลตรวจพื้น — {meta.get("label") or meta["name"]}')
    out = args.out or os.path.join(args.session, 'report_replay.md')
    with open(out, 'w') as f:
        f.write(md)
    rp.figure_png(res, spec, grid, os.path.splitext(out)[0] + '.png')
    print(f'วิเคราะห์ซ้ำเสร็จ: {len(res.defects)} จุด → {out}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='floor_inspect',
                                 description='ตรวจความไม่เรียบของพื้นสำหรับ AMR ด้วย D455')
    sub = ap.add_subparsers(dest='cmd', required=True)

    d = sub.add_parser('demo', help='รันกับสนามทดสอบจำลอง (ไม่ต้องมีกล้อง)')
    from floor_inspect import synth as _synth
    d.add_argument('--scene', default='warehouse', choices=list(_synth.SCENES))
    d.add_argument('--height', type=float, default=0.6)
    d.add_argument('--pitch', type=float, default=45.0)
    d.add_argument('--cell-mm', type=float, default=10.0, dest='cell_mm')
    d.add_argument('--amr', default='amr_generic', choices=['amr_generic', 'amr_small',
                                                            'amr_heavy', 'agv_tugger'])
    d.add_argument('--out', default='./floor_inspect_demo')
    d.set_defaults(func=cmd_demo)

    b = sub.add_parser('bench', help='ตารางความสามารถ: ตรวจเจออะไรได้ที่ระยะไหน')
    b.add_argument('--heights', default='0.4,0.6,0.8')
    b.add_argument('--pitch', type=float, default=45.0)
    b.add_argument('--frames', type=int, default=8)
    b.set_defaults(func=cmd_bench)

    r = sub.add_parser('replay', help='เปิดเซสชันที่บันทึกไว้มาวิเคราะห์ซ้ำ')
    r.add_argument('session')
    r.add_argument('--amr', default='amr_generic')
    r.add_argument('--pothole-mm', type=float, default=None, dest='pothole_mm')
    r.add_argument('--out', default=None)
    r.set_defaults(func=cmd_replay)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
