#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inspect_node — โหนด ROS2 ตรวจสภาพพื้นสด จาก RealSense D455 + แดชบอร์ดเว็บ

เปิดหน้าเว็บที่ http://localhost:8770/ เห็นพร้อมกัน:
  • ภาพกล้องพร้อม overlay ความเสี่ยง (ระบายสีทับพื้นจริงต่อพิกเซล)
  • แผนที่มองจากบน (deviation heat map) + ภาพตัดขวาง
  • ตารางความผิดปกติพร้อมคำตัดสินว่าหุ่นผ่านได้ไหม + ความเร็วปลอดภัย
  • ปุ่มสแกนละเอียด (สะสมหลายเฟรมตอนจอดนิ่ง → เห็นได้ละเอียดกว่าเฟรมเดียว ~2 เท่า)
  • ปุ่มบันทึกเซสชัน (เก็บ DEM + ภาพ + JSON ไว้ทำรายงาน/เป็นชุดข้อมูล annotate ต่อ)

รันคู่กับกล้องที่เปิดไว้แล้ว:
  ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true \
       enable_accel:=true unite_imu_method:=1 pointcloud.enable:=false
  ros2 run floor_inspect inspect_node

พารามิเตอร์ทุกตัวปรับสดได้ระหว่างรัน (อ่านใหม่ทุก tick) เช่น
  ros2 param set /floor_inspect pothole_mm 3.0
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import math
import json
import re
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, Imu
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter as RclParameter, ParameterValue, ParameterType

from floor_inspect import surface_core as sc
from floor_inspect import risk as rk
from floor_inspect import render as rd
from floor_inspect import fnumbers as fn
from floor_inspect import session as ses
from floor_inspect.dashboard_html import HTML_PAGE

STATE_LOCK = threading.Lock()
STATE = {'ready': False, 't': 0.0, 'fps': 0.0, 'ok': False, 'note': 'กำลังรอภาพจากกล้อง…',
         'cam_height_m': None, 'tilt_deg': None, 'plane_rms_mm': None, 'inlier': 0.0,
         'fill': None, 'noise_mm': None, 'rough_mm': None, 'stale_s': None,
         'area_m2': 0.0, 'coverage': 0.0, 'detectable_mm': None, 'n_defects': 0,
         'verdict': '—', 'verdict_msg': '', 'ok_pct': None, 'worst': None,
         'FF': None, 'FL': None, 'FF_class': '', 'gap2m_mm': None,
         'defects': [], 'spec': '', 'mode': 'live', 'scan_progress': None}

FRAME_LOCK = threading.Lock()
FRAMES = {'overlay': None, 'topdown': None, 'profile': None}
# เวลาล่าสุดที่มีคนเปิดดูแต่ละภาพ — ใช้ข้ามการวาดภาพที่ไม่มีใครดู
# (วาดแผนที่มองจากบน + ภาพตัดขวางกินราว 65 ms/เฟรม ซึ่งเป็นเวลาที่เอาไป
#  วิเคราะห์เฟรมถัดไปได้ดีกว่า ถ้าไม่มีใครเปิดหน้าเว็บอยู่)
WATCH_LOCK = threading.Lock()
WATCHERS = {'overlay': 0.0, 'topdown': 0.0, 'profile': 0.0}
WATCH_TTL = 3.0


def watching(key: str) -> bool:
    with WATCH_LOCK:
        return (time.time() - WATCHERS.get(key, 0.0)) < WATCH_TTL


def imgmsg_to_bgr(msg):
    """ROS Image → numpy BGR (เลี่ยง cv_bridge เพราะ opencv-python 5.0 ทำให้พัง)"""
    arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
    if msg.encoding.startswith('rgb'):
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


def depthmsg_to_mm(msg):
    return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width)


def tilt_deg_from_accel(a):
    """มุมก้มของกล้องจากเวกเตอร์ความเร่ง (เฟรม optical: x ขวา, y ลง, z หน้า)

    ตอนกล้องวางระนาบ แรงที่วัดได้ชี้ตาม +y (ลง) ล้วน → ก้ม 0°
    ตอนกล้องมองตรงลงพื้น แรงชี้ตาม −z → ก้ม 90°
    """
    v = np.asarray(a, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-3:
        return None
    v = v / n
    return float(np.degrees(np.arctan2(-v[2], v[1])))


class FloorInspect(Node):
    def __init__(self, out_dir: str, port: int = 8770):
        super().__init__('floor_inspect')
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # ── พารามิเตอร์ (ปรับสดได้ทุกตัว) ──
        d = self.declare_parameter
        d('interval', 0.0)     # 0 = เร็วสุดเท่าที่ CPU ไหว; >0 = จำกัดคาบเป็นวินาที
        d('cell_mm', 10.0)
        d('zmax', 3.0)
        d('base_win_m', 0.40)
        d('pothole_mm', 4.0)
        d('bump_mm', 4.0)
        d('step_mm', 3.0)
        d('k_sigma', 4.0)
        d('k_rough', 3.0)                  # กรองเนื้อผิวพื้น (พรม/ลายกระเบื้อง) แรงแค่ไหน
        d('max_usable_mm', 15.0)           # เกินนี้ถือว่า "วัดไม่ได้แล้ว" ตัดออกจากพื้นที่สำรวจ
        d('min_points_cell', 6.0)          # จุด depth ต่อเซลล์ขั้นต่ำ — ต่ำกว่านี้ตัดออก
                                           # (0 = ปิด) ตัวคุมจุดปลอมที่ระยะไกลตัวหลัก
        d('texture_alpha', 0.30)           # ต่ำ = เกณฑ์นิ่งขึ้น แต่ตามการเปลี่ยนฉากช้าลง
        d('confirm_frames', 5)             # ดูย้อนหลังกี่เฟรมเวลายืนยัน defect
        d('confirm_hits', 3)               # ต้องโผล่กี่เฟรมในนั้นถึงจะรายงาน (1 = ปิด)
                                           # 5/3 วัดได้ sd ของจำนวนจุด 1.15 -> 0.92
                                           # หน่วง ~0.5 s ที่ 9 fps (สิ่งกีดขวางไม่ถูกหน่วง)
        d('confirm_radius_m', 0.10)        # ถือว่าเป็นจุดเดียวกันถ้าอยู่ห่างไม่เกินนี้
        d('min_snr', 3.0)
        d('obstacle_mm', 60.0)
        d('amr_preset', 'amr_generic')
        d('view_mode', 'risk')             # risk | dev
        d('dev_vmax_mm', 15.0)
        d('scan_frames', 12)
        d('profile_y_m', 0.0)
        # ── ค่าตั้งกล้อง: rs_launch.py ไม่รับเป็น launch arg จึงต้องยิงผ่าน service
        #    ทีหลัง ตั้ง -1 = ไม่ยุ่ง ปล่อยค่าที่กล้องเป็นอยู่ ──
        d('laser_power', 150.0)            # 0-360; ลดถ้าจุดสะท้อนบานเป็นรู
        d('ae_limit_us', 20000)            # เพดานเวลาเปิดหน้ากล้องของ depth (-1 = ไม่จำกัด)
        d('hdr', False)                    # สลับสองค่าแสงแล้วรวม — สู้พื้นมันวาว
        d('power_line_hz', 50)             # ไทย 50 Hz (0 = ปิด, -1 = ไม่ยุ่ง)
        d('pixel_stride', 1)               # 2 = ย่อ depth ก่อนวิเคราะห์ (เร็วขึ้น ~2 เท่า)
                                           # ใช้ได้ผลดีเฉพาะตอนภาพ >= 1280x720
        d('motion_compensate', True)       # ชดเชยการเลื่อนของพื้นก่อนจับคู่กับประวัติ
                                           # ปิดได้ถ้ากล้องอยู่กับที่แน่นอนและอยากประหยัด CPU
                                           # แต่ห้ามปิดตอนหุ่นวิ่ง — ตัวยืนยันจะกลบของจริงทิ้ง
        d('max_labels', 8)                 # ป้ายข้อความบนภาพสูงสุดกี่จุด (เรียงตามความเสี่ยง)
                                           # ที่เหลือขึ้นกากบาทอย่างเดียว กัน "ป้ายเต็มจอ"
        d('stale_after_s', 2.0)            # ไม่มีเฟรมใหม่นานกว่านี้ = ประกาศว่ากล้องหยุดส่ง
        d('camera_ns', '/camera/camera')

        self.interval = float(self.get_parameter('interval').value)
        self.next_due = 0.0
        self.depth_seq = 0
        self.done_seq = -1
        self.tex_ab = None                 # (a, b) ของเส้นเนื้อผิว — พาข้ามเฟรมเพื่อความนิ่ง
        self.det_hist = []                 # ประวัติ defect ไม่กี่เฟรมหลัง ใช้ยืนยันก่อนรายงาน
        self.hist_ref = None               # (สูง, ก้ม) ตอนสร้างประวัติ — เปลี่ยนมากแปลว่าย้ายกล้อง
        self.prev_dev = None               # (กริดเบี่ยงเบนเฟรมก่อน, x_max, y_max, cell)
                                           # ใช้วัดว่าพื้นเลื่อนไปเท่าไรตอนหุ่นวิ่ง
        ns = str(self.get_parameter('camera_ns').value)

        self.K = None
        self.depth = None
        self.color = None
        self.accel = None
        self.last_t = time.time()
        self.t0 = time.time()
        self.fps_ema = 0.0
        self.last_rx = 0.0                 # เวลาที่ได้ depth เฟรมล่าสุด
        self.stalled_since = None          # เวลาที่เริ่มประกาศว่ากล้องหยุดส่ง
        self.last_stall_msg = 0.0

        # โหมดสแกน: สะสม DEM หลายเฟรมตอนจอดนิ่ง
        self.scan_acc = None
        self.scan_left = 0
        self.scan_plane = None
        self.scan_T = None
        self.scan_bounds = None
        self.scan_result = None

        self.last_res = None
        self.last_overlay = None
        self.last_spec = None
        self.last_risk = None
        self.lock = threading.Lock()

        self.create_subscription(CameraInfo, f'{ns}/color/camera_info',
                                 self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, f'{ns}/aligned_depth_to_color/image_raw',
                                 self._on_depth, qos_profile_sensor_data)
        self.create_subscription(Image, f'{ns}/color/image_raw',
                                 lambda m: setattr(self, 'color', imgmsg_to_bgr(m)),
                                 qos_profile_sensor_data)
        self.create_subscription(Imu, f'{ns}/accel/sample',
                                 lambda m: setattr(self, 'accel', (
                                     m.linear_acceleration.x, m.linear_acceleration.y,
                                     m.linear_acceleration.z)),
                                 qos_profile_sensor_data)

        self.pub_overlay = self.create_publisher(Image, '~/overlay', 1)
        self.pub_markers = self.create_publisher(MarkerArray, '~/defects', 1)
        self.pub_json = self.create_publisher(String, '~/report_json', 1)

        self.cam_ns = ns
        self.cam_cli = self.create_client(SetParameters, f'{ns}/set_parameters')
        self.cam_applied = False
        # ตั้ง timer ถี่ไว้แล้วคุมคาบจริงในลูป -> เปลี่ยน interval สดได้โดยไม่ต้องรีสตาร์ต
        self.create_timer(0.02, self.tick)
        self.create_timer(2.0, self._apply_camera_settings)
        self.get_logger().info(f'floor_inspect พร้อม — เปิด http://localhost:{port}/  '
                               f'(กล้อง: {ns}, ผลบันทึกที่ {out_dir})')

    def _apply_camera_settings(self):
        """ยิงค่าคุณภาพภาพเข้าโหนดกล้อง

        ทำผ่าน service ไม่ใช่ launch arg เพราะ rs_launch.py รับ launch arg แค่ 79 ตัว
        และ laser_power / auto_exposure_limit / power_line_frequency ไม่อยู่ในนั้น
        (ยิงครั้งเดียวพอ แต่ถ้ากล้องรีสตาร์ตกลางทางจะยิงซ้ำให้เอง)
        """
        if self.cam_applied or self.depth is None:
            return
        if not self.cam_cli.service_is_ready():
            return

        g = self.get_parameter
        want = []
        lp = float(g('laser_power').value)
        if lp >= 0:
            want.append(('depth_module.laser_power', float(lp)))
        ae = int(g('ae_limit_us').value)
        if ae > 0:
            want.append(('depth_module.auto_exposure_limit_toggle', True))
            want.append(('depth_module.auto_exposure_limit', int(ae)))
        plf = int(g('power_line_hz').value)
        if plf >= 0:
            # ค่าในไดรเวอร์: 0=ปิด 1=50Hz 2=60Hz 3=อัตโนมัติ
            want.append(('rgb_camera.power_line_frequency',
                         {0: 0, 50: 1, 60: 2}.get(plf, 3)))
        # ห้าม AE ยืดเวลาเปิดหน้ากล้องจนเฟรมเรตตก — ภาพจะได้ไม่เบลอตอนหุ่นวิ่ง
        want.append(('rgb_camera.auto_exposure_priority', False))
        hdr = bool(g('hdr').value)
        want.append(('depth_module.hdr_enabled', hdr))
        want.append(('hdr_merge.enable', hdr))

        req = SetParameters.Request()
        for name, val in want:
            pv = ParameterValue()
            if isinstance(val, bool):
                pv.type, pv.bool_value = ParameterType.PARAMETER_BOOL, val
            elif isinstance(val, int):
                pv.type, pv.integer_value = ParameterType.PARAMETER_INTEGER, val
            else:
                pv.type, pv.double_value = ParameterType.PARAMETER_DOUBLE, float(val)
            req.parameters.append(RclParameter(name=name, value=pv))

        fut = self.cam_cli.call_async(req)

        def done(f):
            try:
                res = f.result()
            except Exception as e:
                self.get_logger().warn(f'ตั้งค่ากล้องไม่สำเร็จ: {e}')
                return
            bad = [want[i][0] for i, r in enumerate(res.results) if not r.successful]
            if bad:
                self.get_logger().warn('กล้องไม่รับค่า: ' + ', '.join(bad))
            ok = [f'{n}={v}' for (n, v), r in zip(want, res.results) if r.successful]
            self.get_logger().info('ตั้งค่ากล้องแล้ว: ' + ', '.join(ok))
            self.cam_applied = True

        fut.add_done_callback(done)

    def _on_depth(self, m):
        self.depth = depthmsg_to_mm(m).copy()
        self.depth_seq += 1
        self.last_rx = time.time()

    def _on_info(self, m):
        self.K = (float(m.k[0]), float(m.k[4]), float(m.k[2]), float(m.k[5]))

    # ── พารามิเตอร์ → SurfaceParams (อ่านสดทุก tick) ──
    def _params(self) -> sc.SurfaceParams:
        g = self.get_parameter
        return sc.SurfaceParams(
            zmax=float(g('zmax').value),
            cell=float(g('cell_mm').value) / 1000.0,
            base_win_m=float(g('base_win_m').value),
            pothole_mm=float(g('pothole_mm').value),
            bump_mm=float(g('bump_mm').value),
            step_mm=float(g('step_mm').value),
            k_sigma=float(g('k_sigma').value),
            k_rough=float(g('k_rough').value),
            max_usable_mm=float(g('max_usable_mm').value),
            min_points_cell=float(g('min_points_cell').value),
            min_snr=float(g('min_snr').value),
            obstacle_mm=float(g('obstacle_mm').value),
            texture_prior=self.tex_ab,
            texture_alpha=float(g('texture_alpha').value),
            pixel_stride=max(1, int(g('pixel_stride').value)))

    def _spec(self) -> rk.AMRSpec:
        name = str(self.get_parameter('amr_preset').value)
        return rk.PRESETS.get(name, rk.PRESETS['amr_generic'])

    # ───────────────────────────── วงรอบหลัก ─────────────────────────────

    def tick(self):
        now = time.time()
        # กล้องหยุดส่งเฟรม (สาย USB หลุด แบนด์วิดท์ล้ม ไดรเวอร์ค้าง) เกิดขึ้นจริงกับ D455
        # บน USB2 — ถ้าไม่จับไว้ ระบบจะเสิร์ฟภาพกับตัวเลขชุดเดิมค้างไว้เรื่อย ๆ โดยยัง
        # ขึ้น fps เดิม ซึ่งอันตรายกว่าไม่มีภาพ เพราะคนดูไม่มีทางรู้ว่ามันหยุดไปแล้ว
        self._check_stale(now)
        if self.depth is None or self.K is None or self.color is None:
            return
        self.interval = max(0.0, float(self.get_parameter('interval').value))
        if now < self.next_due:
            return
        # กล้องส่งภาพ 10 Hz แต่ลูปหมุนได้เร็วกว่านั้น — ถ้าไม่กันไว้จะวิเคราะห์
        # เฟรมเดิมซ้ำแล้วรายงาน fps เกินจริง กินซีพียูฟรี ๆ
        if self.depth_seq == self.done_seq:
            return
        self.done_seq = self.depth_seq
        self.next_due = now + self.interval
        dt = now - self.last_t
        self.last_t = now
        if dt > 0:
            inst = 1.0 / dt
            self.fps_ema = inst if self.fps_ema == 0 else 0.8 * self.fps_ema + 0.2 * inst

        p = self._params()
        spec = self._spec()
        depth, K, color = self.depth, self.K, self.color

        if self.scan_left > 0:
            res = self._scan_step(depth, K, p)
            if res is None:
                self._publish_state(p, spec, None, None, note='กำลังสแกน…')
                return
        else:
            res = sc.analyse_depth(depth, K, p)

        if not res.ok:
            self._publish_state(p, spec, None, None,
                                note=self._advice(res.note or 'วิเคราะห์ไม่ได้'),
                                fail=res)
            self._draw(color, None, None, None, p, spec)
            return

        if res.texture_ab is not None:
            self.tex_ab = res.texture_ab
        res.defects = self._confirm(res)
        rk.assess_all(res.defects, spec)
        grid = rk.risk_grid(res.dem, res.dev_local_mm, res.steps_mm, res.slope_deg,
                            spec, res.noise_mm, rough_mm=res.rough_mm)
        with self.lock:
            self.last_res, self.last_spec, self.last_risk = res, spec, grid

        self._publish_state(p, spec, res, grid)
        self._draw(color, res, grid, K, p, spec)
        self._publish_ros(res, color, K)

    def _scan_step(self, depth, K, p):
        """สะสมทีละเฟรมระหว่างสแกน — คืน SurfaceResult เมื่อครบ ไม่งั้นคืน None"""
        xyz, valid = sc.deproject_full(depth, K, p.zmin, p.zmax)
        if valid.sum() < 500:
            return None
        if self.scan_plane is None:
            sub = xyz[::p.stride_fit, ::p.stride_fit][valid[::p.stride_fit, ::p.stride_fit]]
            pl = sc.fit_reference_plane(sub, p)
            if not pl.ok:
                self.scan_left = 0
                return None
            self.scan_plane = pl
            self.scan_T = sc.floor_transform(pl)
            self.scan_bounds = None
        pf = sc.transform_points(xyz, self.scan_T)
        dem = sc.build_dem(pf[valid], p, bounds=self.scan_bounds)
        if dem is None:
            return None
        if self.scan_bounds is None:
            self.scan_bounds = (dem.x_max - dem.shape[0] * dem.cell, dem.x_max,
                                dem.y_max - dem.shape[1] * dem.cell, dem.y_max)
        self.scan_acc = sc.accumulate_dem(self.scan_acc, dem)
        self.scan_left -= 1
        with STATE_LOCK:
            STATE['scan_progress'] = self.scan_left
        if self.scan_left > 0:
            return None

        n_used = self.scan_acc['frames']
        dem_avg, _ = sc.finish_accumulation(self.scan_acc, min_frames=max(2, n_used // 3))
        dem_avg = sc.crop_to_surveyed(dem_avg, p.solid_frac, p.solid_win_m)
        noise = sc.local_noise_mm(dem_avg, p.noise_win_m)
        dem_s = sc.median_prefilter(dem_avg, p.median_pre)
        dev_plane, dev_local, _ = sc.deviation_maps(dem_s, p, noise)
        steps = sc.step_map(dem_s, p.step_base_m)
        slope = sc.slope_map(dem_s)
        solid = sc.solid_mask(dem_s, p.solid_win_m, p.solid_frac)
        rough, _ = sc.surface_floor_mm(
            sc.local_roughness_mm(dev_local, dem_s.cell, p.rough_win_m),
            sc.dem_range_m(dem_s, self.scan_T), dem_s.valid,
            prior_ab=self.tex_ab)
        defects = sc.merge_overlapping(sc.detect_defects(
            dem_s, dev_local, steps, noise, p, self.scan_plane.cam_height_m, solid,
            rough_mm=rough, dev_plane_mm=dev_plane))
        res = sc.SurfaceResult(ok=True, plane=self.scan_plane, T_floor_cam=self.scan_T,
                               dem=dem_s, dev_plane_mm=dev_plane, dev_local_mm=dev_local,
                               steps_mm=steps, slope_deg=slope, noise_mm=noise,
                               rough_mm=rough,
                               defects=defects, pts_floor=pf, valid=valid,
                               note=f'สแกนละเอียด {n_used} เฟรม')
        self.scan_acc = None
        self.scan_plane = None
        with STATE_LOCK:
            STATE['scan_progress'] = None
            STATE['mode'] = 'live'
        return res

    def start_scan(self, n: int):
        self.scan_acc = None
        self.scan_plane = None
        self.scan_bounds = None
        self.scan_left = max(2, int(n))
        with STATE_LOCK:
            STATE['mode'] = 'scan'
            STATE['scan_progress'] = self.scan_left

    # เกณฑ์ของ _scene_shift — ไม่เปิดเป็นพารามิเตอร์เพราะปรับมั่วแล้วอันตราย
    #
    # ตอนแรกใช้ค่า response ของ phaseCorrelate เป็นตัวคัด ซึ่งใช้ไม่ได้: มันไม่ได้
    # normalize จึงขึ้นกับสถิติของลายพื้น เทียบข้ามอุปกรณ์ไม่ได้เลย วัดได้ว่า
    #   เฟรมที่วัดถูกในตัวจำลอง  resp = 1.03-1.14
    #   เฟรมที่วัดถูกจากกล้องจริง resp = 0.36-0.79   (พรม, cell 10 mm)
    # เกณฑ์ 0.70 ที่คัดจากข้อมูลตัวจำลองจึงตัดกล้องจริงทิ้งทั้งหมด 0/9 เฟรม
    # = ปิดการชดเชยเงียบ ๆ บนฮาร์ดแวร์จริง ทั้งที่ค่าที่วัดได้แม่นระดับ 2 mm
    #
    # จึงเลิกเชื่อ resp แล้วตรวจตรง ๆ ว่า "เลื่อนตามที่วัดแล้วสองกริดทับกันจริงไหม"
    # ด้วย NCC ซึ่ง normalize ด้วยความแปรปรวนของสัญญาณเอง เทียบข้ามอุปกรณ์ได้
    # วัดจากกล้องจริง 9 คู่เฟรม:
    #   วัดถูก (err <= 2.7 mm)      NCC 0.67-0.86    resp 0.36-0.79
    #   วัดผิด (err 526-598 mm)     NCC <= 0.00      resp 0.02-0.18
    #   เลื่อนผิดตั้งใจ 15 เซลล์     NCC 0.02-0.13
    # แยกขาดโดยไม่มีช่วงทับกัน — ตั้งที่ 0.45 ให้ห่างจากทั้งสองฝั่ง
    SHIFT_MIN_NCC = 0.45
    SHIFT_MAX_M = 0.40         # เลื่อนเกินนี้ต่อเฟรม แปลว่าสองเฟรมแทบไม่ทับกัน เชื่อค่าไม่ได้
    SHIFT_DEAD_M = 0.005       # ต่ำกว่านี้ถือว่าอยู่กับที่ ไม่ต้องขยับประวัติ

    @staticmethod
    def _align_ncc(A, B, s_col, s_row) -> float:
        """เลื่อน A ตามที่วัดได้แล้ววัดว่าทับกับ B ดีแค่ไหน — คืน NaN ถ้าเหลือพื้นที่ไม่พอ"""
        h, w = A.shape
        M = np.float32([[1.0, 0.0, s_col], [0.0, 1.0, s_row]])
        Aw = cv2.warpAffine(A, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        # ตัดขอบที่ถูกดึงมาจากนอกกริดทิ้ง ไม่งั้นศูนย์ที่เติมเข้าไปจะปลอมค่า NCC
        mr, mc = int(abs(s_row)) + 2, int(abs(s_col)) + 2
        if h - 2 * mr < 12 or w - 2 * mc < 12:
            return float('nan')
        a = Aw[mr:h - mr, mc:w - mc].ravel().astype(np.float64)
        b = B[mr:h - mr, mc:w - mc].ravel().astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / den) if den > 1e-12 else float('nan')

    def _scene_shift(self, res):
        """พื้นเลื่อนไปเท่าไรระหว่างเฟรมก่อนกับเฟรมนี้ — คืน (dx, dy, เชื่อได้ไหม) หน่วยเมตร

        ทำไมต้องมี: พิกัด (x, y) ของ defect วัดจาก "ตีนตั้งฉากของกล้องบนพื้น" ไม่ใช่จากโลก
        (ดู floor_transform) พอหุ่นวิ่ง จุดเดิมบนพื้นจึงได้พิกัดใหม่ทุกเฟรม เอาไปจับคู่กับ
        ประวัติตรง ๆ จะไม่มีวันตรงกัน แล้ว _confirm จะกลบ "ของจริง" ทิ้งหมด ซึ่งอันตราย
        กว่าไม่มีตัวยืนยันเลย — วัดจริงด้วยโค้ดตัวนี้: 0.2 m/s ที่ 3.4 fps เลื่อน 5.9 cm
        ต่อเฟรม เกิน confirm_radius_m (10 cm) ภายในสองเฟรม หลุมจริงถูกกลบ 10 จาก 12 เฟรม
        (ด่านเดิมที่ล้างประวัติเมื่อ "ความสูง/มุมก้มเปลี่ยน" ไม่ยิงเลย เพราะวิ่งตรงบนพื้นราบ
         ทั้งสองค่าไม่เปลี่ยน)

        วัดจากตัว DEM เอง ไม่ใช่ IMU เพราะ accel อย่างเดียวหาระยะเลื่อนไม่ได้ (ต้อง
        อินทิเกรตสองชั้น ดริฟต์ทันที) และยังไม่มี odometry จากหุ่นมาให้ ถ้าวันหนึ่งมี /odom
        ควรใช้ค่านั้นแทน แล้วเหลือตัวนี้ไว้เป็นตัวสำรอง
        """
        d = res.dem
        if d is None or res.dev_local_mm is None:
            self.prev_dev = None
            return 0.0, 0.0, False
        cur = np.nan_to_num(np.asarray(res.dev_local_mm, dtype=np.float32), nan=0.0)
        prev, self.prev_dev = self.prev_dev, (cur, d.x_max, d.y_max, d.cell)
        if prev is None or abs(prev[3] - d.cell) > 1e-9:
            return 0.0, 0.0, False
        pa, px, py, cell = prev
        h = min(pa.shape[0], cur.shape[0])
        w = min(pa.shape[1], cur.shape[1])
        if h < 16 or w < 16:
            return 0.0, 0.0, False
        A = np.ascontiguousarray(pa[:h, :w])
        B = np.ascontiguousarray(cur[:h, :w])
        if A.std() < 1e-6 or B.std() < 1e-6:
            return 0.0, 0.0, False          # ไม่มีลายให้จับคู่ — ตอบไม่ได้
        try:
            (s_col, s_row), _ = cv2.phaseCorrelate(
                A.astype(np.float64), B.astype(np.float64),
                cv2.createHanningWindow((w, h), cv2.CV_64F))
            ncc = self._align_ncc(A, B, s_col, s_row)
        except cv2.error:
            return 0.0, 0.0, False
        # สองเฟรมมีจุดเริ่มกริดคนละที่ (x_max/y_max ขยับตามสิ่งที่มองเห็น)
        # ต้องหักส่วนนั้นออกก่อน ถึงจะเหลือ "การเลื่อนของพื้นจริง"
        dx = s_row * cell - (d.x_max - px)
        dy = s_col * cell - (d.y_max - py)
        ok = (bool(np.isfinite([dx, dy, ncc]).all())
              and ncc >= self.SHIFT_MIN_NCC
              and abs(dx) <= self.SHIFT_MAX_M and abs(dy) <= self.SHIFT_MAX_M)
        return float(dx), float(dy), ok

    def _confirm(self, res):
        """รายงานเฉพาะ defect ที่โผล่ซ้ำในหลายเฟรม

        จุดที่ค่าอยู่ตรงเส้นเกณฑ์พอดีจะสลับเข้า-ออกทุกเฟรมตาม noise ที่เหลือ
        ทำให้หน้าจอกะพริบและนับจำนวนไม่ได้ ทั้งที่ฉากไม่ได้เปลี่ยนเลย
        ของจริงจะอยู่ที่เดิมทุกเฟรม ของปลอมจะไม่อยู่ — ใช้ข้อนี้แยก

        พิกัด defect วัดจากตัวกล้อง ไม่ใช่จากโลก เวลาหุ่นวิ่งจุดเดิมจึงเลื่อนพิกัดทุกเฟรม
        ต้องชดเชยก่อนจับคู่ ไม่งั้นตัวยืนยันจะกลบของจริงทิ้งหมด (ดู _scene_shift)
        ถ้าวัดการเลื่อนไม่ได้ = ไม่รู้ว่าอะไรอยู่ตรงไหน → ล้างประวัติแล้วรายงานทุกเฟรม
        ยอมให้กะพริบดีกว่ายอมให้หลุมหาย
        """
        g = self.get_parameter
        n_hist = max(1, int(g('confirm_frames').value))
        need = max(1, int(g('confirm_hits').value))
        rad = float(g('confirm_radius_m').value)
        cur = list(res.defects)

        # ด่านที่ 1: กล้องเงยหรือยกสูงขึ้น — เรขาคณิตเปลี่ยนทั้งชุด จับคู่ต่อไม่ได้
        tilt = tilt_deg_from_accel(self.accel) if self.accel else 0.0
        ref = (round(res.plane.cam_height_m, 2), round(tilt or 0.0, 0))
        if self.hist_ref is not None and (
                abs(ref[0] - self.hist_ref[0]) > 0.02
                or abs(ref[1] - self.hist_ref[1]) > 3.0):
            self.det_hist = []
        self.hist_ref = ref

        if need <= 1:
            self.det_hist = []
            self.prev_dev = None
            return cur

        # ด่านที่ 2: หุ่นวิ่งไปข้างหน้า/ด้านข้าง — ความสูงกับมุมก้มไม่เปลี่ยนเลย
        # ด่านแรกจึงไม่ยิง ต้องวัดการเลื่อนเองแล้วเลื่อนประวัติตามมาให้อยู่เฟรมเดียวกัน
        if bool(g('motion_compensate').value):
            dx, dy, shift_ok = self._scene_shift(res)
            if not shift_ok:
                self.det_hist = []
            elif dx * dx + dy * dy > self.SHIFT_DEAD_M ** 2:
                self.det_hist = [[(k, x - dx, y - dy) for k, x, y in fr]
                                 for fr in self.det_hist]
        else:
            self.prev_dev = None

        past = list(self.det_hist)
        self.det_hist = (self.det_hist + [[(d.kind, d.x, d.y) for d in cur]])[-n_hist:]
        if len(past) < need - 1:
            return cur                     # ประวัติยังไม่พอ อย่าเพิ่งกั๊กข้อมูล

        out = []
        for d in cur:
            hits = 1
            for frame in past:
                if any(k == d.kind
                       and (x - d.x) ** 2 + (y - d.y) ** 2 <= rad * rad
                       for k, x, y in frame):
                    hits += 1
            # สิ่งกีดขวางใหญ่กับรูข้อมูล ไม่ต้องรอยืนยัน — พลาดแล้วอันตรายกว่า
            if hits >= need or d.kind in ('obstacle', 'missing'):
                out.append(d)
        return out

    def _advice(self, note: str) -> str:
        """อย่าแนะนำสิ่งที่เปิดอยู่แล้ว

        surface_core ไม่รู้ว่ากล้องตั้งค่าอะไรไว้ มันเลยแนะนำ 'ลองเปิด HDR' ทุกครั้ง
        ที่ผิวมันวาว ซึ่งพอเปิด HDR ไปแล้วยังไม่หาย ข้อความนั้นกลายเป็นทางตัน
        คนอ่านไปนั่งหาวิธีเปิดสิ่งที่เปิดอยู่ แทนที่จะรู้ว่าต้องเปลี่ยนพื้นหรือมุมกล้อง
        """
        if 'HDR' not in note:
            return note
        if not bool(self.get_parameter('hdr').value):
            return note
        note = re.sub(r'ลองเปิด HDR[^·]*', '', note).rstrip(' —·')
        return (note + ' — เปิด HDR อยู่แล้วแต่ยังไม่พอ แปลว่าผิวนี้สะท้อน IR '
                'เหมือนกระจก จุดเลเซอร์กระเด็นไปทางอื่นแทนที่จะสะท้อนกลับ '
                'stereo เลยไม่มีอะไรให้จับคู่ · เพิ่ม laser_power ไม่ช่วย '
                '(วัดแล้ว 150/250/360 ให้ depth เท่ากันหมด) · ทางที่ได้ผลคือ '
                'ย้ายไปพื้นด้าน เช่น พรม/ปูน/ยาง หรือโรยฝุ่น/แป้งบาง ๆ '
                'ให้ผิวมีอะไรให้เกาะ')

    def _check_stale(self, now: float):
        """เฝ้าดูว่ายังมี depth เฟรมใหม่เข้ามาไหม — ถ้าไม่ ให้ประกาศออกไปให้ชัด"""
        limit = max(0.5, float(self.get_parameter('stale_after_s').value))
        gap = now - self.last_rx if self.last_rx > 0 else 0.0
        if self.last_rx <= 0 or gap <= limit:
            if self.stalled_since is not None:
                self.get_logger().info('กล้องกลับมาส่งภาพแล้ว')
                self.stalled_since = None
            return

        if self.stalled_since is None:
            self.stalled_since = self.last_rx
            self.det_hist = []         # ประวัติข้ามช่วงที่กล้องดับใช้ไม่ได้
            self.prev_dev = None       # กล้องอาจถูกย้ายไปแล้วระหว่างที่ภาพค้าง
        if now - self.last_stall_msg < 1.0:
            return
        self.last_stall_msg = now
        down = now - self.stalled_since
        self.get_logger().warn(f'ไม่ได้รับ depth มา {down:.0f} วินาที')
        note = (f'กล้องหยุดส่งภาพมาแล้ว {down:.0f} วินาที — ภาพและตัวเลขที่เห็นคือของเก่า '
                'ค้างไว้ ไม่ใช่สภาพพื้นตอนนี้ · สาเหตุที่พบบ่อยกับ D455 บน USB2 คือ '
                'แบนด์วิดท์ล้มหรือไดรเวอร์ค้าง แก้ด้วยการรันสคริปต์ใหม่ '
                '(ถ้ายังไม่หาย ให้ถอดเสียบสาย USB แล้วเสียบพอร์ต USB3 สีน้ำเงิน)')
        with STATE_LOCK:
            STATE.update(ready=False, fps=0.0, ok=False, note=note,
                         stale_s=round(down, 1), n_defects=0, defects=[],
                         verdict='—', verdict_msg='')
        self._stall_banner(down)

    def _stall_banner(self, down: float):
        """แปะแถบเตือนทับภาพล่าสุด เพื่อไม่ให้ภาพค้างดูเหมือนภาพสด"""
        base = self.last_overlay if self.last_overlay is not None else self.color
        if base is None:
            return
        img = base.copy()
        h, w = img.shape[:2]
        img[:] = (img * 0.45).astype(img.dtype)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 220), 6)
        rd.draw_texts(img, [
            ((w // 2, h // 2 - 26), 'กล้องหยุดส่งภาพ', 30, (255, 255, 255),
             (0, 0, 190), 'mm'),
            ((w // 2, h // 2 + 18), f'ภาพค้างมาแล้ว {down:.0f} วินาที', 20,
             (255, 235, 235), (30, 32, 38), 'mm'),
        ])
        self._store('overlay', img)

    # ───────────────────────────── ส่งออก ─────────────────────────────

    def _publish_state(self, p, spec, res, grid, note='', fail=None):
        tilt = tilt_deg_from_accel(self.accel) if self.accel else None
        with STATE_LOCK:
            STATE.update(ready=True, t=time.time() - self.t0, fps=round(self.fps_ema, 1),
                         tilt_deg=None if tilt is None else round(tilt, 1),
                         stale_s=None, spec=spec.name)
            if res is None:
                STATE.update(ok=False, note=note, n_defects=0, defects=[],
                             verdict='—', verdict_msg='', area_m2=0.0)
                # ตอนวิเคราะห์ไม่ผ่าน ค่าวินิจฉัยยิ่งสำคัญ — ผู้ใช้ต้องเห็นว่า rms
                # ลดลงต่ำกว่าเกณฑ์หรือยัง ระหว่างค่อย ๆ ขยับกล้องเล็งพื้น
                pl = getattr(fail, 'plane', None)
                ok_pl = pl is not None and getattr(pl, 'ok', False)
                STATE.update(
                    plane_rms_mm=(round(pl.rms_mm, 2)
                                  if ok_pl and math.isfinite(pl.rms_mm) else None),
                    cam_height_m=(round(pl.cam_height_m, 3)
                                  if ok_pl and math.isfinite(pl.cam_height_m) else None),
                    inlier=round(pl.inlier_frac, 3) if ok_pl else 0.0,
                    fill=(round(float(fail.valid.mean()), 3)
                          if fail is not None and getattr(fail, 'valid', None) is not None
                          else None),
                    noise_mm=None, rough_mm=None,
                    detectable_mm=None, coverage=0.0, ok_pct=None, worst=None,
                    FF=None, FL=None, FF_class='', gap2m_mm=None)
                return
            summ = rk.area_summary(grid, res.dem.cell)
            verdict, msg = rk.overall_verdict(summ, res.defects, spec)
            f = fn.f_numbers(res.dem)
            _, gap2 = fn.straightedge_gap_mm(res.dem, 2.0)
            STATE.update(
                ok=True, note=res.note or '',
                cam_height_m=round(res.plane.cam_height_m, 3),
                plane_rms_mm=round(res.plane.rms_mm, 2),
                inlier=round(res.plane.inlier_frac, 3),
                area_m2=round(res.dem.area_m2, 3),
                coverage=round(res.dem.coverage, 3),
                fill=(round(float(res.valid.mean()), 3)
                      if res.valid is not None else None),
                noise_mm=(round(float(np.nanmedian(res.noise_mm)), 2)
                          if res.noise_mm is not None
                          and np.isfinite(res.noise_mm).any() else None),
                rough_mm=(round(float(np.nanmedian(res.rough_mm)), 2)
                          if res.rough_mm is not None
                          and np.isfinite(res.rough_mm).any() else None),
                detectable_mm=round(res.detectable_mm, 2),
                n_defects=len(res.defects),
                verdict=verdict, verdict_msg=msg,
                ok_pct=summ.get('ok_pct'), worst=summ.get('worst'),
                FF=f.FF, FL=f.FL, FF_class=fn.classify_FF(f.FF),
                gap2m_mm=None if not np.isfinite(gap2) else round(gap2, 2),
                defects=[x.to_dict() for x in
                         sorted(res.defects, key=lambda z: -z.risk_score)[:40]])

    def _draw(self, color, res, grid, K, p, spec):
        if res is None:
            frame = color.copy()
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1),
                          (0, 0, 255), 3)
            rd.draw_text(frame, (12, 12), 'ยังหาระนาบพื้นไม่เจอ', 20, (255, 255, 255),
                         (0, 0, 200))
            self.last_overlay = frame
            self._store('overlay', frame)
            return
        mode = str(self.get_parameter('view_mode').value)
        vmax = float(self.get_parameter('dev_vmax_mm').value)
        # วาดเฉพาะภาพที่มีคนเปิดดูอยู่จริง — แต่ต้องตัดสินแยกกันทีละภาพ
        # (เดิม return ทั้งเมธอดตอนไม่มีคนดู overlay ทำให้สตรีม topdown/profile
        #  ตายไปด้วยเมื่อเปิดดูแค่ภาพเดียว)
        if watching('overlay') or self.pub_overlay.get_subscription_count() > 0:
            g = grid if mode == 'risk' else res.dev_local_mm
            ov = rd.overlay_on_color(color, res, g, mode, vmax=vmax,
                                     defects=res.defects, K=K,
                                     max_labels=int(self.get_parameter('max_labels').value))
            rd.draw_text(ov, (10, 8),
                         f'{spec.name} · เห็นได้ละเอียดสุด {res.detectable_mm:.1f} mm · '
                         f'{res.dem.area_m2:.2f} m²', 15, (240, 240, 240), (30, 32, 38))
            self.last_overlay = ov
            self._store('overlay', ov)
        else:
            self.last_overlay = None
        if watching('topdown'):
            self._store('topdown', rd.topdown_map(
                res, mode, vmax=vmax, defects=res.defects, risk_score=grid,
                max_labels=int(self.get_parameter('max_labels').value)))
        if watching('profile'):
            self._store('profile', rd.profile_image(
                res, float(self.get_parameter('profile_y_m').value)))

    def _store(self, key, img):
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            with FRAME_LOCK:
                FRAMES[key] = buf.tobytes()

    def _publish_ros(self, res, color, K):
        ov = self.last_overlay
        if ov is not None and self.pub_overlay.get_subscription_count() > 0:
            m = Image()
            m.header.frame_id = 'camera_color_optical_frame'
            m.header.stamp = self.get_clock().now().to_msg()
            m.height, m.width = ov.shape[:2]
            m.encoding = 'bgr8'
            m.step = m.width * 3
            m.data = ov.tobytes()
            self.pub_overlay.publish(m)

        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        colmap = {'ok': (0.35, 0.8, 0.25), 'watch': (0.95, 0.85, 0.2),
                  'slow': (0.98, 0.55, 0.15), 'block': (0.9, 0.2, 0.2)}
        for i, d in enumerate(res.defects[:60]):
            mk = Marker()
            mk.header.frame_id = 'floor'
            mk.header.stamp = stamp
            mk.ns = 'floor_defects'
            mk.id = i
            mk.type = Marker.CUBE
            mk.action = Marker.ADD
            mk.pose.position.x = float(d.x)
            mk.pose.position.y = float(d.y)
            mk.pose.position.z = 0.0
            mk.pose.orientation.w = 1.0
            mk.scale.x = max(float(d.length_m), 0.02)
            mk.scale.y = max(float(d.width_m), 0.02)
            mk.scale.z = max(abs(float(d.peak_mm)) / 1000.0, 0.003) \
                if np.isfinite(d.peak_mm) else 0.01
            r, g, b = colmap.get(d.risk, (0.6, 0.6, 0.6))
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = r, g, b, 0.75
            mk.lifetime.sec = 2
            ma.markers.append(mk)
        self.pub_markers.publish(ma)

        with STATE_LOCK:
            payload = json.dumps(STATE, ensure_ascii=False)
        self.pub_json.publish(String(data=payload))

    # ───────────────────────────── บันทึกเซสชัน ─────────────────────────────

    def save_session(self, label: str = '') -> dict:
        with self.lock:
            res, spec, grid = self.last_res, self.last_spec, self.last_risk
        if res is None:
            return {'ok': False, 'error': 'ยังไม่มีผลวิเคราะห์ให้บันทึก'}
        return ses.save(self.out_dir, res, spec, grid, self.color, self.K, label=label)


# ───────────────────────────── เว็บเซิร์ฟเวอร์ ─────────────────────────────

class Handler(BaseHTTPRequestHandler):
    node = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/':
            self._send(HTML_PAGE.encode('utf-8'), 'text/html; charset=utf-8')
        elif path == '/state':
            self._sse()
        elif path == '/state.json':
            # ภาพนิ่งของสถานะแบบ JSON ธรรมดา — เอาไว้เช็คจาก curl/สคริปต์
            # (ตัดภาพ base64 ออก เพราะที่ต้องการคือตัวเลข ไม่ใช่รูป)
            with STATE_LOCK:
                snap = {k: v for k, v in STATE.items()
                        if k not in ('overlay', 'topdown', 'profile')}
            self._json(snap)
        elif path in ('/mjpeg', '/mjpeg/topdown', '/mjpeg/profile'):
            key = {'': 'overlay', '/topdown': 'topdown', '/profile': 'profile'}[
                path.replace('/mjpeg', '')]
            self._mjpeg(key)
        elif path == '/sessions':
            self._json(ses.list_sessions(self.node.out_dir) if self.node else [])
        elif path == '/report':
            self._report()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = self.path.split('?')[0]
        body = b'{}'
        n = int(self.headers.get('Content-Length', 0))
        if n:
            body = self.rfile.read(n)
        try:
            payload = json.loads(body or b'{}')
        except Exception:
            payload = {}

        if path == '/scan':
            n_fr = int(payload.get('frames') or
                       (self.node.get_parameter('scan_frames').value if self.node else 12))
            if self.node:
                self.node.start_scan(n_fr)
            self._json({'ok': True, 'frames': n_fr})
        elif path == '/save':
            self._json(self.node.save_session(str(payload.get('label', '')))
                       if self.node else {'ok': False})
        elif path == '/param':
            self._json(self._set_param(payload))
        elif path == '/annotate':
            self._json(ses.annotate(self.node.out_dir, payload) if self.node
                       else {'ok': False})
        else:
            self.send_response(404)
            self.end_headers()

    def _set_param(self, payload):
        """ปรับพารามิเตอร์จากหน้าเว็บ — ใช้ชื่อ/ชนิดเดียวกับ ros2 param set เป๊ะ ๆ"""
        from rclpy.parameter import Parameter
        if not self.node:
            return {'ok': False}
        name = str(payload.get('name', ''))
        val = payload.get('value')
        try:
            cur = self.node.get_parameter(name).value
        except Exception:
            return {'ok': False, 'error': f'ไม่รู้จักพารามิเตอร์ {name}'}
        try:
            if isinstance(cur, bool):
                val = bool(val)
            elif isinstance(cur, int) and not isinstance(cur, bool):
                val = int(val)
            elif isinstance(cur, float):
                val = float(val)
            else:
                val = str(val)
            self.node.set_parameters([Parameter(name, value=val)])
            return {'ok': True, 'name': name, 'value': val}
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def _report(self):
        from floor_inspect import report as rp
        if not self.node:
            self.send_response(503)
            self.end_headers()
            return
        with self.node.lock:
            res, spec, grid = self.node.last_res, self.node.last_spec, self.node.last_risk
        if res is None:
            self._send('ยังไม่มีผลวิเคราะห์'.encode('utf-8'), 'text/plain; charset=utf-8')
            return
        md = rp.markdown_report(res, spec, grid)
        self._send(md.encode('utf-8'), 'text/markdown; charset=utf-8')

    def _send(self, data, ctype):
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj):
        self._send(json.dumps(obj, ensure_ascii=False).encode('utf-8'),
                   'application/json; charset=utf-8')

    def _sse(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        try:
            while True:
                with STATE_LOCK:
                    payload = json.dumps(STATE, ensure_ascii=False)
                self.wfile.write(f'data: {payload}\n\n'.encode('utf-8'))
                self.wfile.flush()
                time.sleep(0.3)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _mjpeg(self, key):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        try:
            while True:
                with WATCH_LOCK:
                    WATCHERS[key] = time.time()
                with FRAME_LOCK:
                    jpg = FRAMES.get(key)
                if jpg:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpg)}\r\n\r\n'.encode())
                    self.wfile.write(jpg + b'\r\n')
                    self.wfile.flush()
                time.sleep(0.2)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    rclpy.init()
    port = int(os.environ.get('FLOOR_INSPECT_PORT', '8770'))
    out_dir = os.path.expanduser(os.environ.get('FLOOR_INSPECT_OUT',
                                                '~/floor_reports'))
    node = FloorInspect(out_dir, port)
    Handler.node = node
    httpd = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
