#!/usr/bin/env bash
# ตัวช่วยเล็งกล้อง — พิมพ์ค่าวินิจฉัยสด ๆ ทุกครึ่งวินาที ระหว่างค่อย ๆ ขยับกล้อง
# เป้าหมาย: rms < 12 mm และ inlier > 0.6 แล้วบรรทัดจะขึ้น "พร้อม"  (Ctrl-C เพื่อออก)
cd "$(dirname "$0")"
while true; do
  curl -s --max-time 2 http://localhost:8770/state.json | python3 _aim_line.py
  sleep 0.5
done
