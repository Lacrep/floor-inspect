# -*- coding: utf-8 -*-
"""ชุดทดสอบท่อวัดพื้น — ทุกข้อเทียบกับ "เฉลยที่รู้เป๊ะ" จากตัวจำลอง

หลายข้อเป็น regression test ของบั๊กที่เจอจริงระหว่างพัฒนา (มีหมายเหตุกำกับ)
ถ้าข้อไหนแดง แปลว่าความแม่นยำของการวัดถอยหลัง ไม่ใช่แค่โค้ดจัดรูปเปลี่ยน

    cd floor-inspect && python3 -m pytest test/ -v
"""
import os

import numpy as np
import pytest

from floor_inspect import surface_core as sc
from floor_inspect import synth
from floor_inspect import fnumbers as fn
from floor_inspect import risk as rk
from floor_inspect import evaluate as ev


# ───────────────────────────── fixtures ─────────────────────────────

@pytest.fixture(scope='module')
def cam():
    return synth.CameraSpec(height_m=0.60, pitch_deg=45.0)


@pytest.fixture(scope='module')
def flat_render(cam):
    scene = synth.FloorScene(features=[], roughness_mm=0.0)
    depth, K, info = synth.render_depth(scene, cam, rng=np.random.default_rng(1))
    return scene, depth, K, info


@pytest.fixture(scope='module')
def targets_render(cam):
    scene = synth.scene_calibration_targets()
    depth, K, info = synth.render_depth(scene, cam, rng=np.random.default_rng(2))
    return scene, depth, K, info


# ───────────────────────── ตัวจำลองต้องถูกต้องก่อน ─────────────────────────

def test_raycast_geometry_matches_height_field(cam):
    """ถ้าตัวจำลองผิด การทดสอบที่เหลือก็ 'ถูกเทียบกับฉากที่ผิด' ซึ่งอันตรายกว่าไม่ทดสอบ

    regression: เดิมใช้วิธีวนแทนค่าซึ่งไม่ลู่เข้าที่ขอบคม — เปลี่ยนเป็น ray marching หาจุดตัดแรก
    """
    scene = synth.scene_calibration_targets()
    _, _, info = synth.render_depth(scene, cam, add_noise=False)
    v = info['valid']
    z_true = scene.height_mm(info['x'][v], info['y'][v]) * 0.001
    err_mm = (info['z'][v] - z_true) * 1000.0
    assert np.sqrt((err_mm ** 2).mean()) < 1.0        # RMS ต่ำกว่า 1 mm
    assert np.percentile(np.abs(err_mm), 99.5) < 5.0  # เกินได้เฉพาะพิกเซลที่คร่อมขอบ


# ───────────────────────────── ระนาบอ้างอิง ─────────────────────────────

def test_plane_fit_recovers_camera_height(flat_render, cam):
    _, depth, K, _ = flat_render
    p = sc.SurfaceParams()
    xyz, valid = sc.deproject_full(depth, K, p.zmin, p.zmax)
    plane = sc.fit_reference_plane(xyz[::4, ::4][valid[::4, ::4]], p)
    assert plane.ok
    assert abs(plane.cam_height_m - cam.height_m) < 0.003     # คลาดไม่เกิน 3 mm
    assert plane.rms_mm < 2.0
    assert plane.n[1] < 0                                      # normal ต้องชี้ "ขึ้น"


def test_floor_frame_puts_plane_at_zero(flat_render):
    _, depth, K, _ = flat_render
    p = sc.SurfaceParams()
    xyz, valid = sc.deproject_full(depth, K, p.zmin, p.zmax)
    plane = sc.fit_reference_plane(xyz[::4, ::4][valid[::4, ::4]], p)
    pf = sc.transform_points(xyz, sc.floor_transform(plane))
    z = pf[valid][:, 2]
    assert abs(float(np.median(z))) < 0.002        # พื้นควรอยู่ที่ z ≈ 0


# ───────────────────────────── DEM ─────────────────────────────

def test_dem_index_roundtrip():
    dem = sc.DEM(z=np.zeros((50, 60), np.float32), count=np.zeros((50, 60), np.int32),
                 spread_mm=np.zeros((50, 60), np.float32), cell=0.01,
                 x_max=1.0, y_max=0.5)
    for r, c in ((0, 0), (10, 20), (49, 59)):
        x, y = dem.xy_of(r, c)
        r2, c2 = dem.rc_of(x, y)
        assert (int(r2), int(c2)) == (r, c)


def test_dem_median_reduces_noise(flat_render):
    """per-cell median ต้องกด noise ลงจริง ไม่ใช่แค่ย้ายข้อมูล"""
    _, depth, K, _ = flat_render
    p = sc.SurfaceParams()
    xyz, valid = sc.deproject_full(depth, K, p.zmin, p.zmax)
    plane = sc.fit_reference_plane(xyz[::4, ::4][valid[::4, ::4]], p)
    pf = sc.transform_points(xyz, sc.floor_transform(plane))
    pts = pf[valid]
    near = pts[(pts[:, 0] > 0.4) & (pts[:, 0] < 0.8)]
    raw_sigma_mm = float(np.std(near[:, 2])) * 1000.0
    dem = sc.build_dem(pts, p)
    r, _ = dem.rc_of(0.6, 0.0)
    band = dem.z[max(0, int(r) - 15):int(r) + 15]
    cell_sigma_mm = float(np.nanstd(band)) * 1000.0
    # ลดได้จริง แต่ห้ามคาดหวัง 1/√n: stereo หา disparity ด้วย block matching หน้าต่าง ~7 px
    # พิกเซลในเซลล์เดียวกันจึงผิดไปทางเดียวกัน median จึงกดได้แค่ ~0.8 เท่า ไม่ใช่ 0.3
    # (ยืนยันกับกล้องจริง: เฉลี่ย 3x3 ลด sd จาก 1.00 เหลือ 0.87 ไม่ใช่ 0.33)
    assert cell_sigma_mm < raw_sigma_mm * 0.9
    assert cell_sigma_mm > raw_sigma_mm * 0.4, \
        'ถ้าลดได้เยอะขนาดนี้ แปลว่าโมเดล noise ในตัวจำลองกลับไปเป็นแบบอิสระต่อพิกเซลแล้ว'


def test_detection_floor_covers_real_scatter(flat_render, cam):
    """เกณฑ์ตรวจจับต้องครอบ "การกระจายจริง" ของพื้นเรียบ ไม่ใช่แค่ค่าที่ตัวประมาณรายงาน

    local_noise_mm วัด noise จากส่วนตกค้างของ median ซึ่ง *ต่ำกว่าความจริง* เสมอเมื่อ
    noise สัมพันธ์กันเชิงพื้นที่ (วัดได้ ~1.7 เท่าบนตัวจำลองที่สมจริง และเคยวัดได้
    หนักกว่านั้นมากบนพรมจริง: รายงาน 0.3 mm ขณะที่ของจริง 11 mm)

    นั่นคือเหตุผลที่ต้องมี rough_mm — ข้อนี้จึงล็อก "เกณฑ์รวมที่ใช้ตัดสินจริง"
    ไม่ใช่ล็อกตัวประมาณตัวเดียวที่รู้อยู่แล้วว่าไม่พอ
    """
    import cv2
    _, depth, K, _ = flat_render
    p = sc.SurfaceParams()
    res = sc.analyse_depth(depth, K, p)
    assert res.ok
    v = res.dem.valid
    zf = np.where(v, res.dem.z * 1000.0, 0).astype(np.float32)
    m = v.astype(np.float32)
    mu = cv2.blur(zf * m, (15, 15)) / np.maximum(cv2.blur(m, (15, 15)), 1e-6)
    true_sigma = float(np.nanstd(np.where(v, zf - mu, np.nan)[v]))
    floor = np.maximum(p.k_sigma * res.noise_mm, p.k_rough * res.rough_mm)
    ratio = float(np.nanmedian(floor[v])) / max(true_sigma, 1e-6)
    assert ratio > 2.5, f'เกณฑ์แค่ {ratio:.1f}x ของ scatter จริง — จะพ่นจุดปลอม'
    assert ratio < 12.0, f'เกณฑ์ {ratio:.1f}x สูงเกินจน defect จริงจะหลุด'


def test_slope_map_exact_on_known_ramp():
    """regression: เดิม normalize Sobel ด้วย Σ|k| ทำให้ความชันเพี้ยนไป 2.84 เท่า"""
    for true_deg in (1.0, 3.0, 10.0):
        H = W = 160
        r, _ = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        z = np.tan(np.radians(true_deg)) * ((H - r) * 0.01)
        dem = sc.DEM(z=z.astype(np.float32), count=np.full((H, W), 9, np.int32),
                     spread_mm=np.zeros((H, W), np.float32), cell=0.01,
                     x_max=H * 0.01, y_max=W * 0.01)
        got = float(np.nanmedian(sc.slope_map(dem, 0.20)[30:-30, 30:-30]))
        assert abs(got - true_deg) < 0.05


# ───────────────────────── การวัดความลึก ─────────────────────────

def test_tophat_does_not_halve_depth():
    """regression: base = (opening+closing)/2 ทำให้ความลึกที่วัดได้เหลือครึ่งเดียว

    ทดสอบตรง ๆ บน DEM สังเคราะห์ที่รู้ความลึกเป๊ะ ไม่ผ่านกล้อง จึงไม่มี noise มากวน
    """
    H = W = 200
    z = np.zeros((H, W), np.float32)
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    hole = (rr - 100) ** 2 + (cc - 100) ** 2 < 8 ** 2       # หลุมรัศมี 8 เซลล์
    z[hole] = -0.010                                         # ลึก 10 mm พอดี
    dem = sc.DEM(z=z, count=np.full((H, W), 9, np.int32),
                 spread_mm=np.zeros((H, W), np.float32), cell=0.01, x_max=2.0, y_max=1.0)
    _, dev, _ = sc.deviation_maps(dem, sc.SurfaceParams())
    assert -11.0 < float(np.nanmin(dev)) < -9.0              # ต้องได้ ~-10 ไม่ใช่ ~-5


def test_no_ghost_between_two_close_defects():
    """regression: SE กว้างกว่าระยะห่างหลุม → รายงาน 'เนิน' ปลอมตรงพื้นดีระหว่างหลุมสองใบ"""
    H, W = 200, 300
    z = np.zeros((H, W), np.float32)
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    for c0 in (100, 128):                                    # ห่างกัน 28 เซลล์ = 0.28 m
        z[(rr - 100) ** 2 + (cc - c0) ** 2 < 7 ** 2] = -0.012
    dem = sc.DEM(z=z, count=np.full((H, W), 9, np.int32),
                 spread_mm=np.zeros((H, W), np.float32), cell=0.01, x_max=2.0, y_max=1.5)
    _, dev, _ = sc.deviation_maps(dem, sc.SurfaceParams())
    assert float(np.nanmax(dev)) < 3.0                       # ห้ามมี "เนิน" โผล่มา


def _surveyed(res, x, y) -> bool:
    """เป้านี้อยู่ในบริเวณที่ระบบ "รับรองว่าวัดได้" หรือไม่"""
    d = res.dem
    r, c = d.rc_of(x, y)
    r, c = int(r), int(c)
    if not (0 <= r < d.shape[0] and 0 <= c < d.shape[1]):
        return False
    k = max(1, int(round(0.03 / d.cell)))
    return bool(d.valid[max(0, r - k):r + k + 1, max(0, c - k):c + k + 1].any())


def test_detects_calibration_targets(targets_render, cam):
    """เป้ามาตรฐาน 5/10/20 mm ที่อยู่ในพื้นที่ที่ระบบรับรอง ต้องเจอครบ คลาดไม่เกิน 2.5 mm

    เป้าแถวไกลสุด (x=1.39 m จากกล้องสูง 0.60 m) ตกอยู่ที่ ~3 จุด depth ต่อเซลล์
    ซึ่งกล้องจริงเชื่อไม่ได้เลย (วัดจากพรมจริง: เซลล์ความหนาแน่นระดับนั้นพ่นหลุมผีลึก
    37 mm ออกมา) ระบบจึงตัดออกจากพื้นที่สำรวจ — ข้อนี้ยอมรับการตัดนั้น แต่บังคับสองอย่าง:
    เป้าที่ยังอยู่ในพื้นที่ต้องเจอครบ และห้ามกุ defect ในบริเวณที่ตัดไปแล้ว
    """
    scene, depth, K, _ = targets_render
    res = sc.analyse_depth(depth, K, sc.SurfaceParams())
    assert res.ok
    checked = 0
    for ft in scene.features:
        if abs(ft.depth_mm) < 5.0:
            continue                                          # 3 mm อยู่ใต้เกณฑ์ default
        if not _surveyed(res, ft.x, ft.y):
            continue
        checked += 1
        near = min((d for d in res.defects if d.kind != 'missing'),
                   key=lambda d: np.hypot(d.x - ft.x, d.y - ft.y), default=None)
        assert near is not None
        dist = np.hypot(near.x - ft.x, near.y - ft.y)
        assert dist < 0.12, f'{ft.label()} หาไม่เจอ (ใกล้สุดห่าง {dist*1000:.0f} mm)'
        assert abs(near.peak_mm - ft.depth_mm) < 2.5, \
            f'{ft.label()} วัดได้ {near.peak_mm:+.1f} mm'
    assert checked >= 4, f'พื้นที่ที่รับรองเหลือน้อยเกินจะทดสอบอะไรได้ ({checked} เป้า)'
    outside = [d for d in res.defects
               if d.kind != 'missing' and not _surveyed(res, d.x, d.y)]
    assert not outside, f'กุ defect นอกพื้นที่ที่รับรอง: {[d.label() for d in outside]}'


def test_flat_floor_has_no_false_positives(cam):
    """พื้นเรียบ (มีความหยาบผิวจริง 0.4 mm) ต้องไม่รายงานอะไรเลย"""
    scene = synth.scene_flat(roughness_mm=0.4)
    depth, K, _ = synth.render_depth(scene, cam, rng=np.random.default_rng(3))
    res = sc.analyse_depth(depth, K, sc.SurfaceParams())
    assert res.ok
    real = [d for d in res.defects if d.kind != 'missing']
    assert len(real) == 0, f'พื้นเรียบแต่รายงาน {len(real)} จุด: ' \
                           f'{[(d.kind, round(d.peak_mm,1)) for d in real]}'


def test_depth_measurement_is_unbiased(cam):
    """bias รวมของความลึกที่วัดได้ต้องใกล้ศูนย์ — ล็อกค่าคงที่ debias ของ p97

    recall วัดเทียบกับ detectable_mm ที่ระบบประกาศเอง ไม่ใช่ตัวเลขคงที่:
    ระบบมีสิทธิ์พลาดเป้าที่เล็กกว่าที่มันบอกว่าเชื่อได้ แต่ห้ามพลาดเป้าที่ใหญ่กว่านั้น
    (ถ้าล็อกเป็นเลขคงที่ เทสต์จะกลายเป็น "ตัวจำลองต้องเก่งเท่าเดิม" แทนที่จะเป็น
    "ระบบต้องทำได้ตามที่มันโฆษณา")
    """
    scene = ev.depth_sweep_scene()
    matches, res, _ = ev.evaluate_scene(scene, cam, sc.SurfaceParams(), seed=5)
    s = ev.summarise(matches)
    floor = res.detectable_mm
    big = [m for m in matches if abs(m.truth_depth_mm) >= floor * 1.5]
    assert big, f'ไม่มีเป้าไหนใหญ่กว่าเกณฑ์ที่ประกาศ ({floor:.1f} mm) เลย'
    missed = [m for m in big if not m.detected]
    assert not missed, ('ประกาศว่าเชื่อได้ถึง %.1f mm แต่พลาด %s'
                       % (floor, [(m.truth_kind, m.truth_depth_mm) for m in missed]))
    assert abs(s['bias_mm']) < 0.6
    assert s['rmse_mm'] < 2.5


@pytest.mark.slow
def test_accumulation_improves_precision(cam):
    """โหมดจอดนิ่งสแกนต้องแม่นกว่าเฟรมเดียวจริง (ไม่ใช่แค่ช้าลง)

    ไม่เทียบ rmse ของเป้าที่จับคู่ได้ เพราะพอสะสมแล้วเป้า *เล็กลง* เข้ามาอยู่ในชุดที่
    จับคู่ได้ด้วย rmse จึงอาจสูงขึ้นทั้งที่ระบบดีขึ้น (วัดได้ 8 เฟรม rmse 0.579 vs
    เฟรมเดียว 0.557 ขณะที่ recall ขึ้นจาก 0.57 เป็น 0.83) — เทียบสองอย่างที่ตรงประเด็น:
    เจอเป้าได้มากขึ้น และเกณฑ์ที่ประกาศเล็กลง
    """
    scene = ev.depth_sweep_scene()
    m1, r1, _ = ev.evaluate_scene(scene, cam, sc.SurfaceParams(), seed=7, n_frames=1)
    m8, r8, _ = ev.evaluate_scene(scene, cam, sc.SurfaceParams(), seed=7, n_frames=8)
    assert ev.summarise(m8)['recall'] > ev.summarise(m1)['recall']
    assert r8.detectable_mm < r1.detectable_mm * 0.75
    assert abs(ev.summarise(m8)['bias_mm']) <= abs(ev.summarise(m1)['bias_mm']) + 0.2


# ───────────────────────────── F-numbers ─────────────────────────────

def test_ff_higher_on_flatter_floor():
    def ff_of(amp_mm):
        H, W = 300, 300
        rr, _ = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        z = (amp_mm * 0.001 * np.sin(2 * np.pi * rr * 0.01 / 0.5)).astype(np.float32)
        dem = sc.DEM(z=z, count=np.full((H, W), 9, np.int32),
                     spread_mm=np.zeros((H, W), np.float32), cell=0.01, x_max=3.0, y_max=1.5)
        return fn.f_numbers(dem).FF
    assert ff_of(0.5) > ff_of(3.0)                # คลื่นเล็กกว่า → FF สูงกว่า


def test_fl_refuses_short_sample_lines():
    """เส้นวัดสั้นกว่า 3.048 m ต้องคืน None พร้อมเหตุผล ห้ามเดาตัวเลขออกมา"""
    H, W = 150, 150                                # 1.5 × 1.5 m
    dem = sc.DEM(z=np.zeros((H, W), np.float32), count=np.full((H, W), 9, np.int32),
                 spread_mm=np.zeros((H, W), np.float32), cell=0.01, x_max=1.5, y_max=0.75)
    r = fn.f_numbers(dem)
    assert r.FL is None
    assert '3.048' in r.note


def test_straightedge_gap_matches_known_dip():
    """แอ่งลึก 6 mm ต้องให้ช่องใต้ไม้บรรทัดใกล้ 6 mm"""
    H, W = 300, 300
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    z = np.zeros((H, W), np.float32)
    z[(rr - 150) ** 2 + (cc - 150) ** 2 < 25 ** 2] = -0.006
    dem = sc.DEM(z=z, count=np.full((H, W), 9, np.int32),
                 spread_mm=np.zeros((H, W), np.float32), cell=0.01, x_max=3.0, y_max=1.5)
    _, worst = fn.straightedge_gap_mm(dem, 1.0)
    assert 5.0 < worst < 7.0


# ───────────────────────────── โมเดลความเสี่ยง ─────────────────────────────

def test_wheel_bridges_narrow_gap():
    """ร่องแคบมาก ล้อต้องคร่อมได้ ไม่ใช่จมเต็มความลึก"""
    deep = rk.wheel_drop_mm(gap_width_m=0.005, depth_mm=50.0, wheel_d_m=0.15)
    assert deep < 0.2                              # ล้อ Ø150 คร่อมร่อง 5 mm แทบไม่รู้สึก
    wide = rk.wheel_drop_mm(gap_width_m=0.30, depth_mm=15.0, wheel_d_m=0.15)
    assert abs(wide - 15.0) < 1e-6                 # กว้างกว่าล้อ → จมเต็ม


def test_narrow_deep_slot_less_risky_than_wide_shallow_dish():
    """หัวใจของโมเดล: ค่ามิลลิเมตรอย่างเดียวตัดสินไม่ได้ ต้องดูความกว้างด้วย"""
    spec = rk.PRESETS['amr_generic']
    slot = sc.Defect('pothole', 1.0, 0.0, 0.30, 0.005, 0.0015, -50.0, -40.0, 20,
                     (0, 0, 30, 1))
    dish = sc.Defect('pothole', 1.0, 0.0, 0.30, 0.30, 0.09, -15.0, -10.0, 900,
                     (0, 0, 30, 30))
    assert rk.assess_defect(slot, spec)[1] < rk.assess_defect(dish, spec)[1]


def test_smaller_caster_is_stricter():
    d = sc.Defect('step', 1.0, 0.0, 0.05, 0.30, 0.015, 6.0, 5.0, 150, (0, 0, 5, 30))
    small = rk.assess_defect(d, rk.PRESETS['amr_small'])[1]
    big = rk.assess_defect(d, rk.PRESETS['agv_tugger'])[1]
    assert small > big


def test_risk_grid_ignores_pure_noise(cam):
    """regression: ไม่หักพื้น noise ทำให้พื้นเรียบขึ้นสีเตือนทั้งผืน (ok_pct ตกจาก 99% เหลือ 48%)"""
    scene = synth.scene_flat(roughness_mm=0.4)
    depth, K, _ = synth.render_depth(scene, cam, rng=np.random.default_rng(4))
    res = sc.analyse_depth(depth, K, sc.SurfaceParams())
    spec = rk.PRESETS['amr_generic']
    g = rk.risk_grid(res.dem, res.dev_local_mm, res.steps_mm, res.slope_deg,
                     spec, res.noise_mm)
    assert rk.area_summary(g, res.dem.cell)['ok_pct'] > 97.0


def test_safe_speed_drops_with_bigger_bump():
    spec = rk.PRESETS['amr_generic']
    assert rk.safe_speed_ms(20.0, 0.1, spec) < rk.safe_speed_ms(3.0, 0.1, spec)


# ───────────────────────────── แสดงผล / รายงาน ─────────────────────────────

def test_render_and_report_smoke(targets_render, cam):
    from floor_inspect import render as rd
    from floor_inspect import report as rp
    scene, depth, K, info = targets_render
    color = synth.render_color(scene, cam, info)
    res = sc.analyse_depth(depth, K, sc.SurfaceParams())
    spec = rk.PRESETS['amr_generic']
    rk.assess_all(res.defects, spec)
    g = rk.risk_grid(res.dem, res.dev_local_mm, res.steps_mm, res.slope_deg,
                     spec, res.noise_mm)
    ov = rd.overlay_on_color(color, res, g, 'risk', defects=res.defects, K=K)
    assert ov.shape == color.shape
    assert rd.topdown_map(res, 'dev', defects=res.defects, risk_score=g).ndim == 3
    assert rd.profile_image(res, 0.18).ndim == 3
    md = rp.markdown_report(res, spec, g)
    assert 'ASTM E1155' in md and 'คำตัดสิน' in md


# ───────────────────────────── ย่อภาพก่อนวิเคราะห์ ─────────────────────────────

def test_pixel_stride_keeps_geometry(targets_render):
    """ย่อ depth ต้องย่อ K ตามให้ถูกสูตร ไม่งั้นความสูงกล้อง/ตำแหน่ง defect จะเพี้ยน

    regression: (cx+0.5)/s-0.5 ไม่ใช่ cx/s — ถ้าใช้ผิดจะเลื่อนไปครึ่งพิกเซลคูณ s
    ซึ่งที่ระยะ 1 m แปลว่าตำแหน่งเพี้ยนไปหลายมิลลิเมตรแบบเงียบ ๆ
    """
    _, depth, K, _ = targets_render
    a = sc.analyse_depth(depth, K, sc.SurfaceParams(pixel_stride=1))
    b = sc.analyse_depth(depth, K, sc.SurfaceParams(pixel_stride=2))
    assert a.ok and b.ok
    assert abs(a.plane.cam_height_m - b.plane.cam_height_m) < 0.003
    # ทิศของระนาบต้องชี้ทางเดียวกัน (มุมต่างกันไม่เกิน 0.3°)
    cos = abs(float(np.dot(a.plane.n, b.plane.n)))
    assert np.degrees(np.arccos(min(1.0, cos))) < 0.3


def test_pixel_stride_2_still_finds_deep_pothole(cam):
    """ย่อได้ แต่ต้องขยายเซลล์ตาม — และต้องยังเจอหลุมลึก

    stride 2 ทำให้จุด depth ต่อเซลล์เหลือ 1/4 ที่ 848x480 เซลล์ 10 mm เหลือ ~3 จุด
    ซึ่งต่ำกว่าเกณฑ์ min_points_cell — ระบบจะตัดบริเวณนั้นทิ้งอย่างถูกต้อง
    (วัดได้: stride2+cell10 เหลือพื้นที่ 0.42 m², cell15 ได้ 1.00, cell20 ได้ 1.69)
    บทเรียนคือ stride กับ cell ต้องไปด้วยกัน ไม่ใช่ปรับตัวเดียว
    """
    scene = synth.FloorScene(
        features=[synth.Feature('pothole', x=0.75, y=0.0, depth_mm=-16.0, radius_m=0.09)],
        roughness_mm=0.3)
    depth, K, _ = synth.render_depth(scene, cam, rng=np.random.default_rng(7))
    res = sc.analyse_depth(depth, K, sc.SurfaceParams(pixel_stride=2, cell=0.015))
    assert res.ok
    hits = [d for d in res.defects if d.kind == 'pothole' and abs(d.peak_mm) > 8.0]
    assert hits, 'stride 2 + cell 15 mm ต้องยังเจอหลุมลึก 16 mm'

    thin = sc.analyse_depth(depth, K, sc.SurfaceParams(pixel_stride=2, cell=0.010))
    assert (not thin.ok) or thin.dem.area_m2 < res.dem.area_m2, \
        'เซลล์เล็กเกินไปสำหรับ stride 2 ต้องเหลือพื้นที่รับรองน้อยลง ไม่ใช่เท่าเดิม'


def test_dem_grid_is_phase_locked(cam):
    """ขอบกริดต้องตกบนพิกัดโลกที่หารด้วย cell ลงตัวเสมอ

    regression ของจริง: ขอบกริดเดิมมาจาก percentile ของกลุ่มจุด ซึ่งขยับตามพิกเซล
    ไกลสุดที่บังเอิญคืน depth มาในเฟรมนั้น วัดกับกล้องจริง 12 เฟรมติดกันได้ x_max
    เลื่อน 1.454-1.465 m (11 mm) แปลว่าทุกเซลล์ถูกสุ่มตัวอย่างใหม่คนละเฟส
    ทุกเฟรม — ผลวิเคราะห์กะพริบทั้งที่กล้องกับพื้นไม่ขยับเลย
    """
    scene = synth.FloorScene(features=[], roughness_mm=0.4)
    seen = []
    for seed in range(6):
        depth, K, _ = synth.render_depth(scene, cam, rng=np.random.default_rng(seed))
        res = sc.analyse_depth(depth, K, sc.SurfaceParams())
        assert res.ok
        seen.append((res.dem.x_max, res.dem.y_max, res.dem.cell))
    for x_max, y_max, cell in seen:
        for v in (x_max, y_max):
            frac = abs(v / cell - round(v / cell))
            assert frac < 1e-6, f'ขอบกริด {v:.4f} ไม่ตกบนคูณของ cell {cell}'


def test_sparse_cells_are_dropped_not_trusted(cam):
    """เซลล์ที่มีจุด depth น้อยเกินไปต้องถูกตัดออกจากพื้นที่สำรวจ

    วัดจากพรมจริง 12 เฟรม: หลุมผีลึก 37 mm นั่งอยู่บนเซลล์ที่มีเฉลี่ย 4.1 จุด
    ขณะที่เนินจริง 24 mm มี 14 จุด — median ของ 2-4 จุดกรอง outlier ไม่ได้เลย
    และ stereo error สัมพันธ์กันเชิงพื้นที่ ทั้งหย่อมจึงพลาดไปทางเดียวกันหมด
    """
    scene = synth.FloorScene(features=[], roughness_mm=0.4)
    depth, K, _ = synth.render_depth(scene, cam, rng=np.random.default_rng(4))
    res = sc.analyse_depth(depth, K, sc.SurfaceParams())
    assert res.ok
    dens = res.dem.count[res.dem.valid]
    assert dens.min() >= 1
    assert float(np.mean(dens < 3.0)) < 0.15, \
        'พื้นที่ที่รับรองยังเต็มไปด้วยเซลล์ข้อมูลบาง'

    off = sc.analyse_depth(depth, K, sc.SurfaceParams(min_points_cell=0.0))
    assert off.ok and off.dem.area_m2 > res.dem.area_m2, \
        'ปิดเกณฑ์แล้วต้องได้พื้นที่มากกว่า — ถ้าเท่ากันแปลว่าเกณฑ์ไม่ได้ทำงาน'


def test_rejects_tiny_flat_patch(cam):
    """regression: แผ่นเรียบผืนจิ๋วใกล้เลนส์ผ่านเกณฑ์ rms ได้ แล้วระบบขึ้น ok=True
    พร้อมพื้นที่ 0.04 m² — ต้องปฏิเสธ ไม่ใช่รายงานว่า 'ตรวจแล้วไม่พบอะไร'"""
    close = synth.CameraSpec(height_m=0.12, pitch_deg=75.0)
    scene = synth.FloorScene(features=[], roughness_mm=0.2)
    depth, K, _ = synth.render_depth(scene, close, rng=np.random.default_rng(11))
    # จำกัดระยะมองให้เหลือผืนจิ๋ว
    res = sc.analyse_depth(depth, K, sc.SurfaceParams(zmax=0.25))
    assert not res.ok
    assert 'น้อยเกินไป' in res.note or 'แทบไม่มีค่า depth' in res.note


# ─────────────── เนื้อผิวของพื้นเอง (พรม/กระเบื้องมีลาย) ───────────────

def test_rough_floor_does_not_flood_false_defects(cam):
    """regression: บนพรมจริง ระบบเคยรายงาน 176 จุดในเฟรมเดียว เพราะเกณฑ์ตายตัว 4 mm
    ต่ำกว่าที่เนื้อผิวเองสั่น — เกณฑ์ต้องยกตามความขรุขระที่วัดได้"""
    scene = synth.FloorScene(features=[], roughness_mm=3.0)   # พรมหยาบ ไม่มี defect จริง
    depth, K, _ = synth.render_depth(scene, cam, rng=np.random.default_rng(21))
    res = sc.analyse_depth(depth, K, sc.SurfaceParams())
    assert res.ok
    real = [d for d in res.defects if d.kind in ('pothole', 'bump', 'step')]
    assert len(real) <= 5, f'พื้นไม่มี defect แต่รายงาน {len(real)} จุด'


def test_real_pothole_survives_rough_floor(cam):
    """แต่ห้ามหูหนวก: หลุมลึกกว่าเนื้อผิวชัด ๆ ต้องยังเจอบนพรมเดียวกัน"""
    scene = synth.FloorScene(
        features=[synth.Feature('pothole', x=0.70, y=0.0, depth_mm=-18.0, radius_m=0.10)],
        roughness_mm=3.0)
    depth, K, _ = synth.render_depth(scene, cam, rng=np.random.default_rng(22))
    res = sc.analyse_depth(depth, K, sc.SurfaceParams())
    assert res.ok
    hits = [d for d in res.defects if d.kind == 'pothole' and abs(d.peak_mm) > 10.0]
    assert hits, 'หลุม 18 mm บนพรมต้องยังเจอ'


def test_surface_floor_grows_with_range():
    """เนื้อผิว+noise ต้องโตตามระยะกำลังสอง ไม่ใช่คงที่ — ตัวประมาณเดิมนิ่งที่ 0.3 mm
    ทุกระยะ ทั้งที่ค่าจริงบนพรมโตจาก 1.9 mm ที่ 0.4 m เป็น 8 mm ที่ 2 m"""
    H, W = 120, 120
    rng_m = np.tile(np.linspace(0.5, 2.5, H, dtype=np.float32)[:, None], (1, W))
    rough = (0.5 + 1.5 * rng_m ** 2).astype(np.float32)       # ค่าจริงที่ซ่อนไว้
    rough = rough + np.random.default_rng(3).normal(0, 0.05, rough.shape).astype(np.float32)
    fit, ab = sc.surface_floor_mm(rough, rng_m, np.ones((H, W), bool))
    near = float(np.median(fit[:10]))     # 0.5 m
    far = float(np.median(fit[-10:]))     # 2.5 m
    assert far > 3.0 * near, f'ไม่โตตามระยะ: ใกล้ {near:.2f} ไกล {far:.2f}'
    assert abs(near - (0.5 + 1.5 * 0.5 ** 2)) < 0.6


def test_glare_dropout_gives_actionable_note(cam):
    """ผิวมันวาวทำให้ depth เป็นรู — ข้อความต้องบอกว่าเป็นเรื่องแสง ไม่ใช่ให้ไปขยับกล้อง
    (สองอย่างนี้แก้คนละทางกัน ถ้าบอกผิดผู้ใช้เสียเวลาไปนั่งเล็งกล้องเปล่า ๆ)"""
    scene = synth.FloorScene(features=[], roughness_mm=0.3)
    depth, K, _ = synth.render_depth(scene, cam, rng=np.random.default_rng(31))
    rng = np.random.default_rng(32)
    holes = rng.random(depth.shape) < 0.75          # จุดสะท้อนกินไป 75% ของภาพ
    depth = depth.copy()
    depth[holes] = 0
    res = sc.analyse_depth(depth, K, sc.SurfaceParams())
    if not res.ok:
        assert ('มันวาว' in res.note or 'สะท้อนแสง' in res.note
                or 'HDR' in res.note), res.note


def test_specular_surface_is_named_not_blamed_on_aiming(cam):
    """ผิวเคลือบเงาแบบกระจก: stereo จับคู่ได้แต่ "เงาของห้อง" ระยะเลยไกลเกินจริง
    ระบบต้องเรียกชื่ออาการนี้ให้ถูก ไม่ใช่บอกให้ไปขยับกล้อง (ซึ่งแก้ไม่ได้)

    ตัวเลขจำลองมาจากที่วัดจริงบนโต๊ะไม้เคลือบเงา: ได้ depth 16% ของภาพ
    ครึ่งหนึ่งของนั้นรายงานระยะเกิน 1.5 m ทั้งที่ผิวอยู่ห่าง 0.5-1.2 m
    """
    scene = synth.FloorScene(features=[], roughness_mm=0.3)
    depth, K, _ = synth.render_depth(scene, cam, rng=np.random.default_rng(41))
    rng = np.random.default_rng(42)
    depth = depth.copy()
    keep = rng.random(depth.shape) < 0.16          # ได้ depth แค่ 16%
    depth[~keep] = 0
    got = keep & (depth > 0)
    mirror = got & (rng.random(depth.shape) < 0.5)  # ครึ่งหนึ่งเป็นเงาสะท้อน
    depth[mirror] = rng.integers(4000, 60000, size=int(mirror.sum()), dtype=np.uint16)

    res = sc.analyse_depth(depth, K, sc.SurfaceParams())
    assert not res.ok
    assert 'กระจก' in res.note, res.note
    assert 'ไม่มีค่ากล้องตัวไหนแก้ได้' in res.note


def test_normal_floor_is_not_called_specular(flat_render):
    """และห้ามเรียกพื้นปกติว่าเป็นกระจก"""
    _, depth, K, _ = flat_render
    res = sc.analyse_depth(depth, K, sc.SurfaceParams())
    assert res.ok and 'กระจก' not in res.note


def test_texture_fit_is_stable_frame_to_frame(cam):
    """regression: ระบบเคยเด้ง — ข้อมูลเข้านิ่งสนิท (fill sd 0.00, rms sd 0.08) แต่
    พื้นที่สำรวจเด้ง 0.08-3.51 m² และจำนวนจุดเด้ง 0-26 เพราะ fit เส้นเนื้อผิวใหม่
    ทุกเฟรมโดยไม่มองเฟรมก่อน"""
    scene = synth.FloorScene(features=[], roughness_mm=2.0)
    prior = None
    areas, counts = [], []
    for i in range(6):
        depth, K, _ = synth.render_depth(scene, cam, rng=np.random.default_rng(50 + i))
        res = sc.analyse_depth(depth, K, sc.SurfaceParams(texture_prior=prior))
        assert res.ok
        prior = res.texture_ab
        areas.append(res.dem.area_m2)
        counts.append(len(res.defects))
    spread = (max(areas) - min(areas)) / max(np.mean(areas), 1e-6)
    assert spread < 0.35, f'พื้นที่แกว่ง {min(areas):.2f}-{max(areas):.2f} m²'
    assert max(counts) - min(counts) <= 6, f'จำนวนจุดแกว่ง {counts}'


def test_wheel_bridges_narrow_texture(cam):
    """ร่องแคบกว่าล้อจะถูกคร่อมข้าม — ไม่ควรถูกนับเป็นหลุมแม้จะลึก
    (δ = R − √(R² − (w/2)²) ; ร่องกว้าง 20 mm ใต้ล้อ Ø75 ล้อจมได้แค่ 1.3 mm)"""
    p = sc.SurfaceParams()
    R = p.wheel_radius_m
    for w_mm, expect in ((20, 1.35), (40, 5.6), (75, 37.5)):
        w = w_mm / 1000.0
        d = 1e3 * (R - np.sqrt(max(R * R - (w / 2) ** 2, 0.0))) if w < 2 * R else 1e3 * R
        assert abs(d - expect) < 0.5, f'w={w_mm} -> {d:.2f} mm'
    assert p.min_effective_mm > 1.35, 'เกณฑ์ต้องกรองร่องกว้าง 20 mm ออกได้'


# ── เทสต์ที่ต้องใช้ข้อมูลจากกล้องจริง ────────────────────────────────────────
# ตัวจำลองสร้างอาการ "ฐานเฉพาะที่ปีนขึ้นไปอยู่บนของ" ไม่ได้ ลองแล้วถึงกล่องกว้าง
# 0.56 m ก็ยังได้ dev_local = dev_plane เป๊ะ เพราะฉากจำลองสะอาด ของถูกมาร์คเป็น
# "ผู้ต้องสงสัย" ครบทั้งก้อน ฐานจึงมาจากพื้นล้วนตามที่ deviation_maps ออกแบบไว้
# ของจริงต่างตรงที่กล่องอยู่ริมเฟรม เห็นแค่ผิวบนบางส่วน และมีเงาบังอยู่ข้างหลัง
# เคยเขียนเทสต์จำลองไว้แล้วพบว่ามันผ่านทั้งที่ปิดการแก้ทิ้ง จึงเปลี่ยนมาใช้เฟรมจริง
_BOX_NPZ = os.path.join(os.path.dirname(__file__), 'data', 'box_355mm.npz')


def _load_box_frame():
    if not os.path.exists(_BOX_NPZ):
        pytest.skip('ไม่มีไฟล์ตัวอย่างจากกล้องจริง')
    z = np.load(_BOX_NPZ)
    return z['depth_mm'], tuple(float(v) for v in z['K'])


def _box_defect(res):
    """กล่องในฉากนี้อยู่ราว (1.56, -1.07) สูงจากพื้น 355 mm (วัดด้วยตลับเมตร ~35 cm)"""
    near = [d for d in res.defects
            if abs(d.x - 1.56) < 0.25 and abs(d.y + 1.07) < 0.25
            and d.kind in ('bump', 'obstacle')]
    return near


def test_real_box_height_is_reported_from_the_floor():
    """กล่องสูง 355 mm ต้องรายงานความสูงจากพื้น ไม่ใช่จากฐานเฉพาะที่

    ก่อนแก้: รายงาน 71-112 mm ตลอด 8 เฟรม (sd 13) ทั้งที่ dev_plane อ่านได้
    353-356 mm นิ่งอยู่แล้ว (sd 1) — ฐานเฉพาะที่ (base_win_m 0.4 m) ปีนขึ้นไป
    อยู่บนกล่อง เฟรมนี้เป็นเฟรมที่อาการชัดที่สุด: dev_plane 356 vs dev_local 76
    """
    depth, K = _load_box_frame()
    p = sc.SurfaceParams(cell=0.01, min_points_cell=6.0, pothole_mm=20.0, bump_mm=4.0)
    res = sc.analyse_depth(depth, K, p)
    assert res.ok, res.note[:100]
    near = _box_defect(res)
    assert near, 'ไม่เจอกล่องเลย'
    tall = max(near, key=lambda d: d.peak_mm)
    assert tall.kind == 'obstacle', f'ต้องเป็น obstacle ไม่ใช่ {tall.kind}'
    assert 300.0 <= tall.peak_mm <= 400.0, \
        f'กล่องสูงจริง ~355 mm แต่รายงาน {tall.peak_mm:.0f} mm'


def test_real_box_is_one_blob_not_split_into_obstacle_plus_bump():
    """กล่องใบเดียวต้องเป็นก้อนเดียว

    เดิมตรงกลางกล่อง dev_local ยุบเกือบศูนย์ ขอบยังสูง กล่องจึงถูกซอยเป็น
    obstacle 71 mm + bump 52 mm คนละก้อน ทั้งที่ทั้งคู่สูงจากพื้น 355 mm เท่ากัน
    """
    depth, K = _load_box_frame()
    p = sc.SurfaceParams(cell=0.01, min_points_cell=6.0, pothole_mm=20.0, bump_mm=4.0)
    res = sc.analyse_depth(depth, K, p)
    assert res.ok
    near = _box_defect(res)
    assert len(near) == 1, ('กล่องใบเดียวถูกซอยเป็น '
                            + ', '.join(f'{d.kind} {d.peak_mm:.0f}mm' for d in near))


@pytest.mark.parametrize('scene_name, max_err_mm', [('asphalt', 4.0), ('worn', 4.0)])
def test_rough_surface_potholes_found_without_ghosts(scene_name, max_err_mm, cam):
    """พื้นขรุขระที่มีหลุมจริง: ต้องเจอทุกหลุม/เนิน และห้ามกุ defect นอกเฉลย

    ยางมะตอยหยาบ ~2 mm และคอนกรีตสึก ~1 mm — เกณฑ์ที่ปรับตามเนื้อผิวต้องไม่ท่วมด้วยจุดปลอม
    และไม่ฆ่าหลุมจริงทิ้ง วัดจริง: คลาดสูงสุด ~3.4 mm (หลุมตื้นวัดต่ำกว่าจริง ไม่ใช่สูงกว่า)
    """
    scene = synth.SCENES[scene_name]()
    depth, K, _ = synth.render_depth(scene, cam)
    res = sc.analyse_depth(depth, K, sc.SurfaceParams())
    assert res.ok, res.note
    for ft in scene.features:
        near = min((d for d in res.defects if d.kind != 'missing'),
                   key=lambda d: np.hypot(d.x - ft.x, d.y - ft.y), default=None)
        assert near is not None, f'{ft.label()} หาไม่เจอเลย'
        assert np.hypot(near.x - ft.x, near.y - ft.y) < 0.12, f'{ft.label()} หาไม่เจอ'
        assert near.kind == ft.kind, f'{ft.label()} ถูกจัดเป็น {near.kind}'
        assert abs(near.peak_mm - ft.depth_mm) < max_err_mm, \
            f'{ft.label()} วัดได้ {near.peak_mm:+.1f} mm'
    ghosts = [d for d in res.defects if d.kind != 'missing' and
              all(np.hypot(d.x - f.x, d.y - f.y) >= 0.12 for f in scene.features)]
    assert not ghosts, f'พบจุดที่ไม่มีในเฉลย: {[(d.kind, round(d.x, 2), round(d.y, 2)) for d in ghosts]}'
    # ข้อจำกัดที่รู้อยู่: หลุมขอบชันทำให้ฐานเฉพาะที่ "ล้น" ที่ขอบ เกิดเนินปลอม ~10 mm ติดขอบหลุมได้ 1 จุด
    # (asphalt: เนิน +10.5 mm ที่ห่างศูนย์กลางหลุม 0.10 m) — ยอมรับได้ไม่เกิน 1 จุดต่อฉาก
    n_real = sum(d.kind != 'missing' for d in res.defects)
    assert n_real <= len(scene.features) + 1, f'defect เกินเฉลยมากกว่า 1 จุด ({n_real} vs {len(scene.features)})'
