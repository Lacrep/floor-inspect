# -*- coding: utf-8 -*-
"""พิมพ์สถานะการเล็งกล้องหนึ่งบรรทัด (ใช้คู่กับ aim_check.sh)"""
import sys, json

try:
    d = json.load(sys.stdin)
except Exception:
    print('ยังต่อโหนดไม่ได้ (http://localhost:8770) — เปิด scripts/run_floor_inspect.sh ก่อน')
    raise SystemExit

def f(v, n=2, u=''):
    return '  —  ' if v is None else format(v, '.%df' % n) + u

h    = f(d.get('cam_height_m'))
tilt = f(d.get('tilt_deg'), 1)
rms  = f(d.get('plane_rms_mm'))

fill = d.get('fill')
fillp = '  —  ' if fill is None else format(fill * 100, '.0f') + '%'

if d.get('ok'):
    print('พร้อม     | สูง %s m | ก้ม %s° | rms %s mm | fill %s | พื้นที่ %s m² | '
          'เนื้อผิว %s mm | เล็กสุด %s mm | จุดผิดปกติ %d'
          % (h, tilt, rms, fillp, f(d.get('area_m2')), f(d.get('rough_mm')),
             f(d.get('detectable_mm')), d.get('n_defects', 0)))
else:
    # fill ต่ำทั้งที่เล็งพื้นอยู่ = แสงสะท้อน/พื้นมันวาวทำให้ depth เป็นรู
    # -> ลอง python3 -m floor_inspect.tune_camera
    print('ยังไม่ผ่าน | สูง %s m | ก้ม %s° | rms %s mm (ต้อง < 12) | '
          'inlier %s (ต้อง > 0.6) | fill %s' % (h, tilt, rms, f(d.get('inlier')), fillp))
