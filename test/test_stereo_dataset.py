# -*- coding: utf-8 -*-
"""ตัวอ่านชุดข้อมูลถนนจริง (point cloud เมตริก) — ตรวจกับ point cloud ที่สร้างจากตัวจำลอง ซึ่งรู้ K จริง

ไม่ต้องดาวน์โหลดชุดข้อมูลจริง (ไฟล์ละ ~40 MB) เพื่อให้เทสต์รันได้ทุกที่
"""
import os

import numpy as np
import pytest
import scipy.io as sio

from floor_inspect import stereo_dataset as sd
from floor_inspect import surface_core as sc
from floor_inspect import synth


@pytest.fixture(scope='module')
def synth_cloud():
    """depth จากตัวจำลอง → point cloud มม. ในกรอบกล้อง (แบบเดียวกับไฟล์ xyzPoints)"""
    scene = synth.SCENES['asphalt']()
    cam = synth.CameraSpec(height_m=0.6, pitch_deg=45.0)
    depth, K, _ = synth.render_depth(scene, cam)
    fx, fy, cx, cy = K
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth.astype(np.float64)
    P = np.dstack([(u - cx) / fx * Z, (v - cy) / fy * Z, Z])
    P[Z <= 0] = np.nan
    return scene, depth, K, P


def test_intrinsics_recovered_from_point_cloud(synth_cloud):
    """ถอด K จากตัวข้อมูลเองต้องได้ใกล้ค่าจริง (คลาด < 0.5%) เพราะชุดข้อมูลจริงไม่ได้ให้ K มา"""
    _, _, K, P = synth_cloud
    depth, K2 = sd.depth_and_intrinsics(P)
    assert depth.shape == P.shape[:2]
    for got, want in zip(K2, K):
        assert abs(got - want) / want < 0.005, f'K ถอดได้ {K2} ต่างจากจริง {K}'


def test_mat_roundtrip_and_pipeline_finds_potholes(synth_cloud, tmp_path):
    """.mat → depth → analyse_depth ต้องเจอหลุมเหมือนป้อน depth ตรง ๆ"""
    scene, _, _, P = synth_cloud
    f = str(tmp_path / 'xyz.mat')
    sio.savemat(f, {'xyzPoints': np.nan_to_num(P, nan=0.0)})
    depth, K = sd.depth_and_intrinsics(sd.load_xyz_mat(f))
    res = sc.analyse_depth(depth, K, sc.SurfaceParams())
    assert res.ok, res.note
    holes = [d for d in res.defects if d.kind == 'pothole']
    assert len(holes) >= 4, f'ควรเจอหลุมครบ 4 แต่ได้ {len(holes)}'


def test_load_rejects_wrong_variable(tmp_path):
    f = str(tmp_path / 'bad.mat')
    sio.savemat(f, {'foo': np.zeros((4, 4, 3))})
    with pytest.raises(ValueError, match='xyzPoints'):
        sd.load_xyz_mat(f)


def test_label_agreement_perfect_when_label_is_truth(synth_cloud):
    """ให้หน้ากากป้ายเป็น 'พื้นที่ที่ลึกกว่าฐานจริง' ของระบบเอง → IoU ต้องใกล้ 1 (เช็คการแมปพิกเซล↔เซลล์)"""
    _, depth, K, _ = synth_cloud
    res = sc.analyse_depth(depth, K, sc.SurfaceParams())
    # สร้างป้ายจาก DEM กลับเป็นพิกเซล โดยใช้ฟังก์ชันเดียวกับที่ตรวจ แล้วเทียบกันเอง
    fx, fy, cx, cy = K
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float64)
    ok = z > 0
    pts = np.c_[((uu - cx) / fx * z)[ok], ((vv - cy) / fy * z)[ok], z[ok]] / 1000.0
    pf = sc.transform_points(pts, res.T_floor_cam)
    r, c = res.dem.rc_of(pf[:, 0], pf[:, 1])
    r, c = np.floor(r).astype(int), np.floor(c).astype(int)
    h, w = res.dem.z.shape
    inb = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    lab = np.zeros((H, W), bool)
    idx = np.flatnonzero(ok.ravel())[inb]
    lab.ravel()[idx] = (res.dev_local_mm <= -4.0)[r[inb], c[inb]]
    ag = sd.label_agreement(lab, depth, K, res, step=2)
    assert ag['depressed']['iou'] > 0.9, ag
