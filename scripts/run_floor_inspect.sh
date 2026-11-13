#!/usr/bin/env bash
# เปิดระบบตรวจพื้น AMR (กล้อง D455 + โหนดวิเคราะห์ + หน้าเว็บ 8770) ในคำสั่งเดียว
#   ./run_floor_inspect.sh                        ค่าเริ่มต้น (balanced, amr_generic)
#   ./run_floor_inspect.sh detail                 โหมดละเอียด (เล็กสุด 1.6 mm)
#   ./run_floor_inspect.sh smooth amr_small       โหมดลื่น + สเปกหุ่นเล็ก
#   GLARE=1 ./run_floor_inspect.sh detail         เปิด HDR สู้พื้นมันวาว
#   CARPET=1 ./run_floor_inspect.sh detail        พื้นพรม — ยกเกณฑ์หลุมกันหลุมผี
#   POTHOLE=15 BUMP=8 ./run_floor_inspect.sh      ตั้งเกณฑ์เองทีละค่า (mm)
#   ROS_WS=~/my_ws ./run_floor_inspect.sh         ระบุ workspace ที่ build ไว้ (ค่าเริ่มต้น ~/ros2_ws)
set -e

QUALITY="${1:-balanced}"

# เกณฑ์ความลึก/ความสูงที่ถือว่าเป็น defect (mm)
#   CARPET=1 ยกเกณฑ์ "หลุม" อย่างเดียวเป็น 20 mm ส่วน "เนิน" คงไว้ที่ 4 mm
#   เพราะบนพรม stereo อ่านเป็นหลุมผีถึง 18 จุด/เฟรม (หางของการกระจายไปถึง
#   -12.5 mm บนพื้นเปล่า ๆ) แต่ของที่วางอยู่จริงยังต้องเจอที่ 4 mm เหมือนเดิม
if [ "${CARPET:-0}" = "1" ]; then
  POTHOLE="${POTHOLE:-20.0}"; BUMP="${BUMP:-4.0}"
else
  POTHOLE="${POTHOLE:-4.0}"; BUMP="${BUMP:-}"
fi
AMR="${2:-amr_generic}"
GLARE="${GLARE:-false}"
LASER="${LASER:-150}"
AE_LIMIT="${AE_LIMIT:-20000}"

case "$QUALITY" in
  smooth|balanced|detail) ;;
  amr_*|agv_*) AMR="$QUALITY"; QUALITY="balanced" ;;   # เผื่อพิมพ์สเปกหุ่นมาเป็นตัวแรกแบบเดิม
  *) echo "quality ต้องเป็น smooth | balanced | detail (ได้รับ '$QUALITY')"; exit 1 ;;
esac

echo "── ตรวจกล้องก่อน ──"
if ! lsusb | grep -qi '8086:0b5c\|Intel'; then
  echo "ไม่พบ RealSense บนพอร์ต USB"
  echo "  • เสียบสาย USB-C เข้าพอร์ต USB 3 (ช่องสีฟ้า/มี SS) แล้วรันใหม่"
  echo "  • ถ้าเสียบแล้วยังไม่ขึ้น ลองสลับสาย — สาย charge-only ใช้ไม่ได้"
  exit 1
fi
rs-enumerate-devices -s || true

USBVER=$(rs-enumerate-devices 2>/dev/null | grep -i 'usb type' | head -1)
echo "$USBVER"
if echo "$USBVER" | grep -q '2\.'; then
  echo "  ⚠ ต่ออยู่แบบ USB 2 — จะได้แค่ 640x480@15 (librealsense ลดให้เอง)"
  echo "    ท่อวิเคราะห์ทำงานได้ปกติ แต่พื้นที่ต่อเฟรมแคบลงและ fps ต่ำลง"
fi

echo "── ล้างอินสแตนซ์เก่า ──"
pkill -9 -f inspect_node  2>/dev/null || true
pkill -9 -f realsense2_camera_node 2>/dev/null || true
sleep 1

source "/opt/ros/${ROS_DISTRO_NAME:-humble}/setup.bash"
# ROS_WS = โฟลเดอร์ workspace ที่ colcon build ไว้ (ค่าเริ่มต้น ~/ros2_ws)
source "${ROS_WS:-$HOME/ros2_ws}/install/setup.bash"

echo "── เปิดระบบ ──"
echo "   โหมดภาพ : $QUALITY   (smooth = ลื่น | balanced = สมดุล | detail = ละเอียด)"
echo "   สเปกหุ่น: $AMR"
echo "   HDR     : $GLARE     laser $LASER   ae_limit ${AE_LIMIT}us"
echo "   เกณฑ์   : หลุม ${POTHOLE}mm  เนิน ${BUMP:-$POTHOLE}mm"
echo "   หน้าเว็บ: http://localhost:8770/"
exec ros2 launch floor_inspect inspect.launch.py \
     quality:="$QUALITY" amr:="$AMR" glare:="$GLARE" \
     laser_power:="$LASER" ae_limit_us:="$AE_LIMIT" \
     pothole_mm:="$POTHOLE" bump_mm:="$BUMP"
