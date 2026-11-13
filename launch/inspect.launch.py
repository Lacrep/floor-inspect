# -*- coding: utf-8 -*-
"""เปิดกล้อง D455 + โหนดตรวจพื้น พร้อมกันในคำสั่งเดียว

    ros2 launch floor_inspect inspect.launch.py quality:=detail amr:=amr_small

โหมดภาพ (quality) — ตัวเลขมาจากการวัดจริงบนสาย USB 2 ของเครื่องนี้:

    smooth    depth 640x480x30  + color 640x480x30   สตรีม 30 Hz  วิเคราะห์ ~6.9 fps
              เล็กสุด 3.9 mm · พื้นที่ 2.6 m²        → ตอนหุ่นวิ่ง ต้องการภาพลื่น
    balanced  depth 848x480x10  + color 1280x720x10  สตรีม 9.1 Hz วิเคราะห์ ~8.7 fps
              เล็กสุด 2.5 mm · พื้นที่ 3.0 m²        → ค่าเริ่มต้น ดีสุดโดยรวม
    detail    depth 1280x720x5  + color 1280x720x5   สตรีม 4.4 Hz วิเคราะห์ ~4.7 fps
              เล็กสุด 1.6 mm · พื้นที่ 3.5 m²        → จอดนิ่งแล้วสแกนละเอียด

ถ้าย้ายไปเสียบ USB 3 ค่าพวกนี้จะดีขึ้นมาก (848x480 ได้ถึง 90 Hz, 1280x720 ได้ 30 Hz)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# (depth_profile, color_profile, pixel_stride)
QUALITY = {
    'smooth':   ('640x480x30',  '640x480x30',  1),
    'balanced': ('848x480x10',  '1280x720x10', 2),
    'detail':   ('1280x720x5',  '1280x720x5',  1),
}


def _setup(context, *args, **kwargs):
    def cfg(name):
        return LaunchConfiguration(name).perform(context)

    q = cfg('quality')
    if q not in QUALITY:
        raise RuntimeError('quality ต้องเป็นหนึ่งใน %s (ได้รับ "%s")'
                           % (' | '.join(QUALITY), q))
    depth_profile, color_profile, stride = QUALITY[q]
    if cfg('pixel_stride'):
        stride = int(cfg('pixel_stride'))

    glare = cfg('glare').lower() in ('1', 'true', 'yes', 'on')

    cam_args = {
        'align_depth.enable': 'true',
        'pointcloud.enable': 'false',
        'enable_accel': 'true',
        'enable_gyro': 'false',
        'unite_imu_method': '1',
        'depth_module.depth_profile': depth_profile,
        'rgb_camera.color_profile': color_profile,

        'rgb_camera.enable_auto_exposure': 'true',
        'depth_module.enable_auto_exposure': 'true',
    }
    # หมายเหตุ: laser_power / auto_exposure_limit / power_line_frequency /
    # auto_exposure_priority ไม่ได้อยู่ในรายการ launch arg 79 ตัวของ rs_launch.py
    # (ใส่ไปก็โดนเมิน พร้อมพ่น Warning) inspect_node จึงยิงเข้าไปทาง service
    # /camera/camera/set_parameters ให้แทน หลังภาพแรกมาถึง

    if glare:
        # HDR: กล้องสลับยิงสองค่าแสงแล้วรวมกัน — ตัวช่วยที่ตรงจุดที่สุดสำหรับพื้นมันวาว
        # (เฟรมสว่างเก็บมุมมืด เฟรมมืดเก็บจุดสะท้อนที่ปกติจะไหม้ขาว)
        # แลกมาด้วยอัตราเฟรมที่ได้จริงลดลงครึ่งหนึ่ง
        cam_args.update({
            'depth_module.exposure.1': '8500',
            'depth_module.gain.1': '16',
            'depth_module.exposure.2': '100',
            'depth_module.gain.2': '16',
        })

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])),
        condition=IfCondition(LaunchConfiguration('start_camera')),
        launch_arguments=cam_args.items())

    node = Node(
        package='floor_inspect', executable='inspect_node', name='floor_inspect',
        output='screen',
        parameters=[{
            'amr_preset': cfg('amr'),
            'cell_mm': float(cfg('cell_mm')),
            'pothole_mm': float(cfg('pothole_mm')),
            # แยกจาก pothole_mm ได้ เพราะพื้นบางชนิดต้องยกเกณฑ์หลุมอย่างเดียว:
            # บนพรม stereo อ่านเป็นหลุมผีเยอะ แต่ของที่วางอยู่ (สายไฟ/อะแดปเตอร์)
            # ยังต้องเจอที่ 4 mm เหมือนเดิม — ถ้าผูกสองค่าไว้ด้วยกันจะต้องเสียอย่างหนึ่ง
            'bump_mm': float(cfg('bump_mm') or cfg('pothole_mm')),
            'zmax': float(cfg('zmax')),
            'pixel_stride': int(stride),
            'interval': float(cfg('interval')),
            'camera_ns': cfg('camera_ns'),
            'laser_power': float(cfg('laser_power')),
            'ae_limit_us': int(cfg('ae_limit_us')),
            'hdr': bool(glare),
            'power_line_hz': int(cfg('power_line_hz')),
        }])
    return [camera, node]


def generate_launch_description():
    args = [
        DeclareLaunchArgument('quality', default_value='balanced',
                              description='smooth | balanced | detail'),
        DeclareLaunchArgument('glare', default_value='false',
                              description='true = เปิด HDR สู้พื้นมันวาว/แสงสะท้อน'),
        DeclareLaunchArgument('laser_power', default_value='150',
                              description='0-360; ลดลงถ้าจุดสะท้อนบานเป็นรู, เพิ่มถ้าพื้นด้านมืด'),
        DeclareLaunchArgument('ae_limit_us', default_value='20000',
                              description='เพดานเวลาเปิดหน้ากล้องของ depth (ไมโครวินาที)'),
        DeclareLaunchArgument('pixel_stride', default_value='',
                              description='ทับค่าของ quality (1 = ละเอียดสุด, 2 = เร็วขึ้นเท่าตัว)'),
        DeclareLaunchArgument('power_line_hz', default_value='50',
                              description='ความถี่ไฟบ้าน 50 (ไทย) | 60 | 0 ปิด | -1 ไม่ยุ่ง'),
        DeclareLaunchArgument('interval', default_value='0.0',
                              description='0 = วิเคราะห์เร็วสุดเท่าที่ CPU ไหว'),
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('amr', default_value='amr_generic',
                              description='amr_generic | amr_small | amr_heavy | agv_tugger'),
        DeclareLaunchArgument('cell_mm', default_value='10.0'),
        DeclareLaunchArgument('pothole_mm', default_value='4.0'),
        DeclareLaunchArgument('bump_mm', default_value='',
                              description='ว่าง = ใช้ค่าเดียวกับ pothole_mm'),
        DeclareLaunchArgument('zmax', default_value='3.0'),
        DeclareLaunchArgument('camera_ns', default_value='/camera/camera'),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_setup)])
