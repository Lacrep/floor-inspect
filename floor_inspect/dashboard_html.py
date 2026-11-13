# -*- coding: utf-8 -*-
"""หน้าเว็บแดชบอร์ดของ inspect_node (แยกไฟล์ไว้ให้โหนดอ่านง่าย)"""

HTML_PAGE = r"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ตรวจสภาพพื้นสำหรับ AMR — D455</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; padding:20px; background:#0b0f14; color:#e8edf2;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans Thai",sans-serif; }
  h1 { font-size:17px; font-weight:600; color:#9fb3c8; margin:0 0 4px; letter-spacing:.02em; }
  .conn { font-size:12px; color:#55677c; margin-bottom:16px; }
  .conn.live { color:#4ade80; }
  .wrap { display:grid; grid-template-columns: 1.4fr 1fr; gap:16px; align-items:start; }
  @media (max-width:1100px){ .wrap{ grid-template-columns:1fr; } }
  .card { background:#131a22; border:1px solid #202b38; border-radius:14px; padding:16px; }
  .card h2 { font-size:12px; color:#7c8ea3; text-transform:uppercase; letter-spacing:.08em;
             margin:0 0 12px; font-weight:600; }
  img.view { width:100%; border-radius:10px; display:block; background:#000; }
  .verdict { display:flex; align-items:center; gap:14px; padding:14px 16px; border-radius:12px;
             margin-bottom:14px; border:1px solid transparent; }
  .verdict .tag { font-size:22px; font-weight:800; padding:6px 16px; border-radius:999px; white-space:nowrap; }
  .verdict .msg { font-size:13.5px; line-height:1.5; }
  .v-OK    { background:#0f2418; border-color:#1e5a34; } .v-OK .tag{ background:#14321f; color:#4ade80; }
  .v-WATCH { background:#2a2410; border-color:#6b5a23; } .v-WATCH .tag{ background:#3a3216; color:#facc15; }
  .v-SLOW  { background:#2e1d0d; border-color:#7a4a18; } .v-SLOW .tag{ background:#3d2611; color:#fb923c; }
  .v-BLOCK { background:#2c1212; border-color:#6b2323; } .v-BLOCK .tag{ background:#3a1616; color:#f87171; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; }
  .stat { background:#0f151d; border:1px solid #1b2530; border-radius:10px; padding:10px 12px; }
  .stat .l { font-size:10.5px; color:#7c8ea3; text-transform:uppercase; letter-spacing:.06em; }
  .stat .v { font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; margin-top:3px; }
  .stat .u { font-size:12px; color:#7c8ea3; font-weight:500; margin-left:3px; }
  .stat .s { font-size:10.5px; color:#55677c; margin-top:4px; line-height:1.35; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th,td { text-align:left; padding:7px 8px; border-bottom:1px solid #1b2530; font-variant-numeric:tabular-nums; }
  th { color:#7c8ea3; font-weight:600; text-transform:uppercase; font-size:10px; letter-spacing:.05em; }
  .pill { padding:2px 9px; border-radius:999px; font-size:11px; font-weight:700; white-space:nowrap; }
  .p-ok{background:#14321f;color:#4ade80} .p-watch{background:#3a3216;color:#facc15}
  .p-slow{background:#3d2611;color:#fb923c} .p-block{background:#3a1616;color:#f87171}
  .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom:12px; }
  button { background:#1e2936; color:#e8edf2; border:1px solid #2b3949; border-radius:8px;
           padding:9px 15px; font-size:13px; font-weight:600; cursor:pointer; font-family:inherit; }
  button:hover { background:#26364a; border-color:#3b82f6; }
  button.primary { background:#2563eb; border-color:#2563eb; color:#fff; }
  button.primary:hover { background:#1d4ed8; }
  input,select { background:#0f151d; border:1px solid #202b38; color:#e8edf2; border-radius:8px;
                 padding:8px 10px; font-size:13px; font-family:inherit; }
  .muted { color:#55677c; font-size:11.5px; line-height:1.5; margin-top:8px; }
  .tabs { display:flex; gap:6px; margin-bottom:10px; }
  .tabs button { padding:6px 13px; font-size:12px; }
  .tabs button.on { background:#2563eb; border-color:#2563eb; color:#fff; }
  .reason { font-size:11.5px; color:#8fa3b8; padding:2px 8px 9px; line-height:1.5; }
  .scan { color:#facc15; font-weight:700; }
</style>
</head>
<body>
<h1>ตรวจสภาพพื้นสำหรับ AMR — RealSense D455</h1>
<div class="conn" id="conn">กำลังเชื่อมต่อ…</div>

<div class="verdict v-OK" id="verdict">
  <div class="tag" id="v-tag">—</div>
  <div class="msg" id="v-msg">กำลังรอข้อมูลจากกล้อง…</div>
</div>

<div class="wrap">
  <div>
    <div class="card" style="margin-bottom:16px;">
      <h2>ภาพสด — ระบายสีทับพื้นตามระดับความเสี่ยง</h2>
      <div class="tabs">
        <button id="t-risk" class="on" onclick="setView('risk')">ความเสี่ยงต่อหุ่น</button>
        <button id="t-dev" onclick="setView('dev')">ค่าเบี่ยงเบน (mm)</button>
      </div>
      <img class="view" id="live" src="/mjpeg">
      <div class="muted">เขียว = พื้นที่ตรวจแล้วผ่าน · เหลือง = เฝ้าระวัง · ส้ม = ควรชะลอ ·
        แดง = ผ่านไม่ได้ · เทา (ไม่ย้อมสี) = กล้องมองไม่ถึง/ข้อมูลไม่พอ</div>
    </div>

    <div class="card" style="margin-bottom:16px;">
      <h2>แผนที่มองจากบน</h2>
      <img class="view" id="topdown" src="/mjpeg/topdown">
    </div>

    <div class="card">
      <h2>ภาพตัดขวาง</h2>
      <img class="view" id="profile" src="/mjpeg/profile">
      <div class="row" style="margin:10px 0 0;">
        <label class="muted">ตำแหน่ง y (m)</label>
        <input type="number" id="py" step="0.05" value="0" style="width:90px">
        <button onclick="setParam('profile_y_m', Number(document.getElementById('py').value))">ย้ายเส้นตัด</button>
      </div>
    </div>
  </div>

  <div>
    <div class="card" style="margin-bottom:16px;">
      <h2>สถานะการวัด</h2>
      <div class="grid">
        <div class="stat"><div class="l">เล็กสุดที่ยืนยันได้</div><div class="v" id="s-det">—<span class="u">mm</span></div>
          <div class="s" id="s-det-why">ต่ำกว่านี้แยกจาก noise ไม่ได้</div></div>
        <div class="stat"><div class="l">พื้นที่สำรวจ</div><div class="v" id="s-area">—<span class="u">m²</span></div>
          <div class="s" id="s-cov">ความครอบคลุม —</div></div>
        <div class="stat"><div class="l">ความสูงกล้อง</div><div class="v" id="s-h">—<span class="u">m</span></div>
          <div class="s" id="s-tilt">ก้ม —°</div></div>
        <div class="stat"><div class="l">พื้นที่ผ่านเกณฑ์</div><div class="v" id="s-ok">—<span class="u">%</span></div>
          <div class="s" id="s-worst">แย่สุด —</div></div>
        <div class="stat"><div class="l">FF (ASTM E1155)</div><div class="v" id="s-ff">—</div>
          <div class="s" id="s-ffc">—</div></div>
        <div class="stat"><div class="l">ช่องใต้ไม้บรรทัด 2 m</div><div class="v" id="s-gap">—<span class="u">mm</span></div>
          <div class="s">เกณฑ์หน้างาน TR34/ACI</div></div>
      </div>
      <div class="muted" id="s-note"></div>
    </div>

    <div class="card" style="margin-bottom:16px;">
      <h2>ควบคุม</h2>
      <div class="row">
        <button class="primary" onclick="scan()">🔍 สแกนละเอียด (จอดนิ่ง)</button>
        <span id="scan-st" class="scan"></span>
      </div>
      <div class="row">
        <input type="text" id="label" placeholder="ชื่อจุดที่วัด เช่น aisle-A-01" style="flex:1;min-width:140px">
        <button onclick="save()">💾 บันทึกเซสชัน</button>
        <button onclick="window.open('/report','_blank')">📄 รายงาน</button>
      </div>
      <div class="row">
        <label class="muted">หุ่น</label>
        <select id="spec" onchange="setParam('amr_preset', this.value)">
          <option value="amr_generic">AMR ทั่วไป (caster Ø75)</option>
          <option value="amr_small">AMR เล็ก/ส่งของ (Ø50)</option>
          <option value="amr_heavy">AMR ขนของหนัก (Ø100)</option>
          <option value="agv_tugger">AGV ลากจูง (Ø125)</option>
        </select>
      </div>
      <div class="row">
        <label class="muted">เกณฑ์หลุม/เนิน (mm)</label>
        <input type="number" id="pth" step="0.5" value="4" style="width:80px">
        <button onclick="var v=Number(document.getElementById('pth').value); setParam('pothole_mm',v); setParam('bump_mm',v);">ตั้ง</button>
      </div>
      <div class="muted">ปรับสดได้ทาง ROS ด้วย: <code>ros2 param set /floor_inspect &lt;ชื่อ&gt; &lt;ค่า&gt;</code></div>
      <div class="muted" id="save-st"></div>
    </div>

    <div class="card">
      <h2>ความผิดปกติที่พบ (<span id="n-def">0</span>)</h2>
      <table>
        <thead><tr><th>ชนิด</th><th>x,y (m)</th><th>ขนาด</th><th>ค่า</th><th>ระดับ</th><th>ปลอดภัยที่</th><th></th></tr></thead>
        <tbody id="tbody"></tbody>
      </table>
      <div class="muted" id="empty">ยังไม่พบความผิดปกติในพื้นที่ที่มองเห็น</div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const KIND = {pothole:'หลุม', bump:'เนิน', step:'ลิ่ม/ขอบ', obstacle:'สิ่งกีดขวาง', missing:'ไม่มีข้อมูล'};
const RISK = {ok:'ปกติ', watch:'เฝ้าระวัง', slow:'ชะลอ', block:'ผ่านไม่ได้'};
let view = 'risk';

const f = (n,d,dflt='—') => (n===null||n===undefined||Number.isNaN(n)) ? dflt : Number(n).toFixed(d);

function setView(v){ view=v; $('t-risk').className = v==='risk'?'on':''; $('t-dev').className = v==='dev'?'on':'';
  setParam('view_mode', v); }

async function setParam(name, value){
  await fetch('/param',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name,value})});
}
async function scan(){ await fetch('/scan',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}); }
async function save(){
  $('save-st').textContent='กำลังบันทึก…';
  const r = await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({label:$('label').value})});
  const j = await r.json();
  $('save-st').textContent = j.ok ? `บันทึกแล้ว: ${j.name} (${j.n_defects} จุด, ${j.area_m2} m²)`
                                  : ('บันทึกไม่สำเร็จ: ' + (j.error||''));
}
async function annotate(idx, verdict){
  await fetch('/annotate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({defect_idx:idx, verdict:verdict, by:'dashboard'})});
  $('save-st').textContent = 'บันทึกป้ายกำกับแล้ว (' + verdict + ')';
}

function render(s){
  const vd = $('verdict');
  vd.className = 'verdict v-' + (s.verdict||'OK');
  $('v-tag').textContent = s.verdict || '—';
  $('v-msg').textContent = s.verdict_msg || s.note || '';
  $('s-det').innerHTML = f(s.detectable_mm,1) + '<span class="u">mm</span>';
  // บอกด้วยว่าอะไรเป็นตัวจำกัด: noise ของเซนเซอร์ หรือเนื้อผิวของพื้นเอง
  $('s-det-why').textContent = (s.rough_mm != null && s.noise_mm != null)
      ? (s.rough_mm * 2.5 > s.noise_mm * 4
           ? 'ถูกจำกัดด้วยเนื้อผิวพื้น ' + s.rough_mm.toFixed(1) + ' mm (พรม/ลายกระเบื้อง)'
           : 'ถูกจำกัดด้วย noise เซนเซอร์ ' + s.noise_mm.toFixed(2) + ' mm')
      : 'ต่ำกว่านี้แยกจาก noise ไม่ได้';
  $('s-area').innerHTML = f(s.area_m2,2) + '<span class="u">m²</span>';
  $('s-cov').textContent = 'ความครอบคลุม ' + f(s.coverage*100,0) + '%';
  $('s-h').innerHTML = f(s.cam_height_m,2) + '<span class="u">m</span>';
  $('s-tilt').textContent = 'ก้ม ' + f(s.tilt_deg,1) + '° · fit rms ' + f(s.plane_rms_mm,2) + ' mm';
  $('s-ok').innerHTML = f(s.ok_pct,1) + '<span class="u">%</span>';
  $('s-worst').textContent = 'คะแนนแย่สุด ' + f(s.worst,2);
  $('s-ff').textContent = f(s.FF,1);
  $('s-ffc').textContent = s.FF_class || '—';
  $('s-gap').innerHTML = f(s.gap2m_mm,2) + '<span class="u">mm</span>';
  $('s-note').textContent = s.note || '';
  $('scan-st').textContent = (s.scan_progress!==null && s.scan_progress!==undefined)
      ? ('กำลังสแกน… เหลือ ' + s.scan_progress + ' เฟรม (อย่าขยับกล้อง)') : '';

  const ds = s.defects || [];
  $('n-def').textContent = ds.length;
  $('empty').style.display = ds.length ? 'none' : 'block';
  $('tbody').innerHTML = ds.map((d,i) => `<tr>
      <td>${KIND[d.kind]||d.kind}</td>
      <td>${f(d.x,2)}, ${d.y>=0?'+':''}${f(d.y,2)}</td>
      <td>${f(d.length_m*1000,0)}×${f(d.width_m*1000,0)}</td>
      <td>${d.peak_mm===null||Number.isNaN(d.peak_mm)?'—':(d.peak_mm>=0?'+':'')+f(d.peak_mm,1)+' mm'}</td>
      <td><span class="pill p-${d.risk}">${RISK[d.risk]||d.risk}</span></td>
      <td>${f(d.safe_speed_ms,2)} m/s</td>
      <td><button style="padding:2px 7px;font-size:11px" onclick="annotate(${i},'tp')" title="ยืนยันว่ามีจริง">✓</button>
          <button style="padding:2px 7px;font-size:11px" onclick="annotate(${i},'fp')" title="ตรวจผิด ไม่มีจริง">✕</button></td>
    </tr>
    <tr><td colspan="7" class="reason">${d.reason||''}</td></tr>`).join('');
}

new EventSource('/state').onmessage = e => {
  $('conn').textContent = 'เชื่อมต่ออยู่'; $('conn').className = 'conn live';
  render(JSON.parse(e.data));
};
</script>
</body>
</html>
"""
