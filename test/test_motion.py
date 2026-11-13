"""ตัวยืนยัน defect ต้องยังทำงานตอนกล้องเคลื่อนที่

ที่ต้องมีไฟล์นี้: พิกัด defect วัดจากตัวกล้อง ไม่ใช่จากโลก (surface_core.floor_transform)
เวลาหุ่นวิ่ง จุดเดิมบนพื้นจึงเลื่อนพิกัดทุกเฟรม ถ้าไม่ชดเชยก่อนจับคู่กับประวัติ
_confirm จะกลบ "ของจริง" ทิ้ง — และกลบเฉพาะตอนหุ่นวิ่ง ซึ่งเป็นสภาพเดียวที่ระบบมีไว้ใช้
เคยเป็นแบบนั้นจริง วัดได้ 2 จาก 14 เฟรมที่ 0.2 m/s

จำลองการเลื่อนกล้องด้วยการเลื่อนฉากสวนทาง — สมมูลกันทุกประการ เพราะ synth
วางกล้องไว้ที่จุดกำเนิดเสมอ

การเรนเดอร์แพงกว่าตรรกะที่ทดสอบมาก จึงเรนเดอร์ลำดับเฟรมครั้งเดียวแล้ว cache ไว้
ให้ทุกเทสต์ใช้ร่วมกัน (เปิด/ปิดการชดเชยใช้เฟรมชุดเดียวกันเป๊ะ ซึ่งเทียบกันได้ตรงกว่าด้วย)
"""
import numpy as np
import pytest

import floor_inspect.surface_core as sc
import floor_inspect.synth as sy
import floor_inspect.inspect_node as N


class _Param:
    def __init__(self, v):
        self.value = v


class FakeNode:
    """ผูกเมธอดตัวจริงจาก FloorInspect เข้ากับ state ขั้นต่ำ — ไม่ต้องมี ROS ทำงานอยู่"""
    SHIFT_MIN_NCC = N.FloorInspect.SHIFT_MIN_NCC
    SHIFT_MAX_M = N.FloorInspect.SHIFT_MAX_M
    SHIFT_DEAD_M = N.FloorInspect.SHIFT_DEAD_M
    # ต้องห่อ staticmethod กลับ ไม่งั้น Python ผูกเป็น method ธรรมดาแล้วยัด self เข้าไปด้วย
    _align_ncc = staticmethod(N.FloorInspect._align_ncc)
    _scene_shift = N.FloorInspect._scene_shift
    _confirm = N.FloorInspect._confirm

    def __init__(self, compensate=True, hits=3, frames=5, radius=0.10):
        self.det_hist, self.hist_ref, self.prev_dev, self.accel = [], None, None, None
        self._pv = dict(confirm_frames=frames, confirm_hits=hits,
                        confirm_radius_m=radius, motion_compensate=compensate)

    def get_parameter(self, k):
        return _Param(self._pv[k])


# ความละเอียดต้องสมจริงพอ ไม่งั้นจะไปทดสอบสภาพที่ของจริงไม่เคยเจอ:
# ที่ 424x240 cell 30mm กริดมีลายน้อยจนวัดการเลื่อนไม่ได้เลย (resp ต่ำ err เป็นร้อย mm)
# ส่วน 848x480 cell 20mm ให้ resp ~1.1 err ~3 mm ซึ่งใกล้ของจริง (1280x720 cell 10-20mm)
CAM = sy.CameraSpec(width=848, height=480, height_m=0.60, pitch_deg=45.0)
PARAMS = sc.SurfaceParams(cell=0.02, pothole_mm=8.0, bump_mm=8.0)
X0, RAD, STEP, NFRAMES = 1.20, 0.12, 0.08, 7
_CACHE = {}


def _sequence(step=STEP, seed=5):
    """ลำดับผลวิเคราะห์ของหลุมจริงหนึ่งจุดที่กล้องค่อย ๆ วิ่งเข้าหา — คืน [(x จริง, res)]"""
    key = (round(step, 4), seed)
    if key in _CACHE:
        return _CACHE[key]
    rng = np.random.default_rng(seed)
    out = []
    for i in range(NFRAMES):
        x = X0 - i * step
        scene = sy.FloorScene(
            features=[sy.Feature('pothole', x=x, y=0.0, depth_mm=-22.0, radius_m=0.10)],
            roughness_mm=0.5)
        depth, K, _ = sy.render_depth(scene, CAM, rng=rng)
        res = sc.analyse_depth(depth, K, PARAMS)
        if res.ok:
            out.append((x, res))
    _CACHE[key] = out
    return out


def _replay(seq, compensate):
    """คืน (รายงานกี่เฟรม, เฟรมที่ตรวจเจอตั้งแต่ต้นกี่เฟรม)"""
    node = FakeNode(compensate=compensate)
    reported = detected = 0
    for x, res in seq:
        def near(ds):
            return any(abs(d.x - x) < RAD and abs(d.y) < RAD for d in ds)
        if not near(res.defects):
            continue          # เฟรมนี้ตรวจไม่เจอตั้งแต่แรก ไม่ใช่เรื่องของตัวยืนยัน
        detected += 1
        if near(node._confirm(res)):
            reported += 1
    return reported, detected


def test_moving_camera_does_not_hide_real_defects():
    """0.08 m/เฟรม = 0.27 m/s ที่ 3.4 fps — เกิน confirm_radius_m (10 cm) ภายในสองเฟรม"""
    rep, det = _replay(_sequence(), compensate=True)
    assert det >= 4, f'ฉากไม่ได้ทดสอบอะไรเลย (ตรวจเจอ {det} เฟรม)'
    assert rep == det, f'กลบของจริงไป {det - rep}/{det} เฟรมทั้งที่เปิดการชดเชย'


def test_motion_compensation_is_what_makes_the_difference():
    """กันไม่ให้เทสต์ข้างบนผ่านด้วยเหตุบังเอิญ — ปิดการชดเชยแล้วต้องพังให้เห็น"""
    seq = _sequence()
    off, det = _replay(seq, compensate=False)
    on, _ = _replay(seq, compensate=True)
    assert det >= 4
    assert off < det * 0.6, f'ปิดการชดเชยแล้วยังรายงาน {off}/{det} — เทสต์นี้ไม่ได้วัดอะไร'
    assert on > off


def test_static_camera_keeps_confirming():
    """การชดเชยต้องไม่ไปทำลายกรณีกล้องอยู่กับที่ ซึ่งเป็นเหตุผลที่มีตัวยืนยันแต่แรก"""
    rep, det = _replay(_sequence(step=0.0, seed=9), compensate=True)
    assert det >= 4
    assert rep == det


def test_shift_estimate_matches_the_real_distance():
    """ค่าที่วัดได้ต้องใกล้ระยะจริง ไม่ใช่แค่ "ไม่ศูนย์" — ถ้าเพี้ยน ประวัติจะถูกเลื่อน
    ไปผิดที่ ซึ่งแย่กว่าไม่เลื่อนเลย เกณฑ์ 30 mm = 1/3 ของ confirm_radius_m"""
    node = FakeNode()
    errs = []
    for i, (_, res) in enumerate(_sequence()):
        dx, dy, ok = node._scene_shift(res)
        if i == 0:
            assert not ok, 'เฟรมแรกยังไม่มีอะไรให้เทียบ ต้องบอกว่าเชื่อไม่ได้'
            continue
        if not ok:
            continue            # ปฏิเสธได้ ตราบใดที่ค่าที่รับมาแม่น (ทิศที่ปลอดภัย)
        errs.append(abs(dx - STEP))
        assert abs(dy) < 0.04, f'เฟรม {i}: ไม่ได้เลื่อนด้านข้างแต่วัดได้ {dy:+.3f} m'
    assert len(errs) >= 3, f'รับมาแค่ {len(errs)} เฟรม — เกณฑ์ NCC เข้มไปจนใช้งานไม่ได้'
    assert float(np.median(errs)) < 0.03, f'คลาดเคลื่อน {np.median(errs) * 1000:.0f} mm'


def test_unreliable_shift_falls_back_to_reporting_everything():
    """วัดไม่ได้ = ไม่รู้ว่าอะไรอยู่ตรงไหน ต้องล้างประวัติแล้วรายงานทุกเฟรม
    ทิศที่ปลอดภัยคือยอมกะพริบ ไม่ใช่ยอมให้หลุมหาย"""
    seq = _sequence()
    if not seq:
        pytest.skip('ฉากนี้วิเคราะห์ไม่ผ่าน')
    _, res = seq[0]
    node = FakeNode()
    for _ in range(4):
        node._confirm(res)
    assert node.det_hist, 'ควรมีประวัติสะสมไว้แล้ว'
    # ยัดกริดเฟรมก่อนที่ไม่มีลายและอยู่คนละที่ → phase correlation ต้องบอกว่าเชื่อไม่ได้
    node.prev_dev = (np.zeros_like(res.dev_local_mm, dtype=np.float32),
                     res.dem.x_max + 99.0, res.dem.y_max, res.dem.cell)
    out = node._confirm(res)
    assert node.det_hist == [[(d.kind, d.x, d.y) for d in res.defects]], \
        'ประวัติเก่าต้องถูกล้าง เหลือแค่เฟรมปัจจุบัน'
    assert len(out) == len(res.defects), 'ตอนไม่แน่ใจต้องรายงานทุกอย่าง ไม่ใช่กั๊ก'
