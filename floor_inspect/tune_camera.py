# -*- coding: utf-8 -*-
"""กวาดค่าตั้งกล้องแล้ววัดผลจริง แทนการเดา

ปัญหาพื้นมันวาว/แสงสะท้อนไม่มีค่าตายตัวที่ถูกเสมอ — ขึ้นกับผิวพื้น มุมไฟ และมุมกล้อง
ของหน้างานนั้น ๆ สคริปต์นี้จึงลองทีละค่าแล้ววัดสามอย่างที่บอก "ภาพลึกดีขึ้นจริงไหม":

    fill      สัดส่วนพิกเซลที่ได้ค่า depth   ยิ่งสูงยิ่งดี (แสงสะท้อนทำให้เป็นรู -> fill ตก)
    rms       ความกระจายของระนาบที่ fit ได้  ยิ่งต่ำยิ่งดี
    noise     สัญญาณรบกวนต่อเซลล์ (mm)       ยิ่งต่ำยิ่งดี

คะแนนรวม = fill / (1 + noise)  — ให้ค่าที่ "เห็นพื้นได้ครบและนิ่ง" ชนะ

ใช้:  python3 -m floor_inspect.tune_camera            กวาดชุดมาตรฐาน
      python3 -m floor_inspect.tune_camera --apply    กวาดแล้วตั้งค่าที่ดีที่สุดค้างไว้
ต้องเปิด run_floor_inspect.sh ค้างไว้ก่อน (ทั้งกล้องและโหนดต้องรันอยู่)
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request

CAM = '/camera/camera'
STATE_URL = 'http://localhost:8770/state.json'


def set_param(name, value):
    """ros2 param set เช็คชนิดเป๊ะ

    laser_power เป็น double — ถ้าส่งเลขไม่มีจุดทศนิยม จะถูกอ่านเป็น integer
    แล้วโดนปฏิเสธเงียบ ๆ ต้องส่งแบบมีจุดเสมอ
    """
    if value is True:
        kind = 'true'
    elif value is False:
        kind = 'false'
    elif isinstance(value, float):
        kind = repr(float(value))
    else:
        kind = str(value)
    r = subprocess.run(['ros2', 'param', 'set', CAM, name, kind],
                       capture_output=True, text=True, timeout=20)
    return 'Set parameter successful' in (r.stdout + r.stderr)


def read_state(timeout=3.0):
    with urllib.request.urlopen(STATE_URL, timeout=timeout) as f:
        return json.loads(f.read().decode('utf-8'))


def measure(settle=2.5, n=6, gap=0.4):
    """รอให้ค่าใหม่มีผล แล้วเก็บหลายเฟรมเฉลี่ย — เฟรมเดียวแกว่งเกินกว่าจะตัดสินได้"""
    time.sleep(settle)
    fills, rmss, noises = [], [], []
    for _ in range(n):
        try:
            s = read_state()
        except Exception:
            time.sleep(gap)
            continue
        if s.get('fill') is not None:
            fills.append(s['fill'])
        if s.get('plane_rms_mm') is not None:
            rmss.append(s['plane_rms_mm'])
        if s.get('noise_mm') is not None:
            noises.append(s['noise_mm'])
        time.sleep(gap)

    def med(a):
        return None if not a else sorted(a)[len(a) // 2]
    return med(fills), med(rmss), med(noises)


def score(fill, noise):
    if fill is None:
        return -1.0
    return fill / (1.0 + (noise if noise is not None else 0.0))


def row(label, fill, rms, noise):
    def f(v, n=3):
        return '  —  ' if v is None else format(v, '.%df' % n)
    print('  %-28s fill %s | rms %s mm | noise %s mm | คะแนน %s'
          % (label, f(fill), f(rms, 2), f(noise, 2), f(score(fill, noise), 3)))


def sweep(name, values, fmt=None):
    print('\n── กวาด %s ──' % name)
    best, results = None, []
    for v in values:
        if not set_param(name, v):
            print('  ตั้ง %s = %s ไม่สำเร็จ (ข้าม)' % (name, v))
            continue
        fill, rms, noise = measure()
        label = (fmt % v) if fmt else '%s = %s' % (name.split('.')[-1], v)
        row(label, fill, rms, noise)
        sc = score(fill, noise)
        results.append((sc, v, fill, rms, noise))
        if best is None or sc > best[0]:
            best = (sc, v, fill, rms, noise)
    return best, results


def main(argv=None):
    ap = argparse.ArgumentParser(prog='tune_camera',
                                 description='กวาดค่าตั้งกล้องแล้ววัดผลจริง')
    ap.add_argument('--apply', action='store_true',
                    help='ตั้งค่าที่ดีที่สุดค้างไว้เมื่อกวาดเสร็จ')
    ap.add_argument('--quick', action='store_true', help='กวาดชุดสั้น')
    a = ap.parse_args(argv)

    try:
        s = read_state()
    except Exception as e:
        print('ต่อโหนดไม่ได้ (%s) — เปิด scripts/run_floor_inspect.sh ก่อน' % e)
        return 1
    if not s.get('ready'):
        print('โหนดยังไม่ได้รับภาพจากกล้อง')
        return 1
    if not s.get('ok'):
        print('เตือน: ตอนนี้ระนาบยังไม่ผ่านเกณฑ์ (%s)' % (s.get('note') or ''))
        print('       ค่า rms/noise จะยังไม่มีความหมาย — ควรเล็งกล้องลงพื้นก่อนกวาด')
        print('       (fill ยังใช้เทียบกันได้อยู่)\n')

    print('ค่าตั้งต้น:')
    row('ก่อนปรับ', s.get('fill'), s.get('plane_rms_mm'), s.get('noise_mm'))

    # ไดรเวอร์รับเฉพาะทวีคูณของ 30 (ros2 param describe บอก Step: 30.0)
    # ค่าที่ไม่ลงตัวจะถูกปฏิเสธเงียบ ๆ
    laser_vals = ([60.0, 150.0, 240.0] if a.quick
                  else [0.0, 60.0, 90.0, 150.0, 210.0, 240.0, 300.0, 360.0])
    ae_vals = [8000, 20000] if a.quick else [4000, 8000, 20000, 50000, 200000]

    best_laser, _ = sweep('depth_module.laser_power', laser_vals,
                          fmt='laser_power = %.0f')
    if best_laser:
        set_param('depth_module.laser_power', best_laser[1])
        print('  -> เลือก laser_power = %.0f' % best_laser[1])

    set_param('depth_module.auto_exposure_limit_toggle', True)
    best_ae, _ = sweep('depth_module.auto_exposure_limit', ae_vals,
                       fmt='ae_limit = %s us')
    if best_ae:
        set_param('depth_module.auto_exposure_limit', best_ae[1])
        print('  -> เลือก ae_limit = %s us' % best_ae[1])

    print('\n── ลอง HDR (สลับสองค่าแสงแล้วรวม — ตัวช่วยหลักของพื้นมันวาว) ──')
    hdr_rows = []
    for on in (False, True):
        set_param('depth_module.hdr_enabled', on)
        set_param('hdr_merge.enable', on)
        fill, rms, noise = measure(settle=4.0)
        row('HDR ' + ('เปิด' if on else 'ปิด'), fill, rms, noise)
        hdr_rows.append((score(fill, noise), on))
    best_hdr = max(hdr_rows)[1]
    set_param('depth_module.hdr_enabled', best_hdr)
    set_param('hdr_merge.enable', best_hdr)
    print('  -> เลือก HDR = %s' % ('เปิด' if best_hdr else 'ปิด'))

    fill, rms, noise = measure(settle=3.0, n=8)
    print('\nผลรวมหลังปรับ:')
    row('หลังปรับ', fill, rms, noise)
    print('\nค่าที่ได้:')
    print('  laser_power        = %s'
          % (('%.0f' % best_laser[1]) if best_laser else '(ไม่เปลี่ยน)'))
    print('  auto_exposure_limit= %s us' % (best_ae[1] if best_ae else '(ไม่เปลี่ยน)'))
    print('  hdr                = %s' % ('เปิด' if best_hdr else 'ปิด'))

    if a.apply:
        print('\nตั้งค้างไว้ในกล้องที่รันอยู่แล้ว — ใส่ลง launch ถาวรด้วย:')
        print('  ros2 launch floor_inspect inspect.launch.py \\')
        print('       laser_power:=%.0f ae_limit_us:=%s glare:=%s'
              % (best_laser[1] if best_laser else 150.0,
                 best_ae[1] if best_ae else 20000,
                 'true' if best_hdr else 'false'))
    else:
        print('\n(ค่าเหล่านี้ตั้งค้างในกล้องที่รันอยู่ แต่จะหายเมื่อรีสตาร์ต '
              'ใส่เป็น argument ของ launch ถ้าจะใช้ถาวร)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
