# Third-party notices

## Stereo pothole datasets (ภาพถนนจริงใน README)

ภาพถนนและข้อมูลความลึกที่ใช้ในหัวข้อ "ถนนขรุขระจริงที่มีหลุมจริง" ของ [README](../README.md) มาจากชุดข้อมูล
**stereo pothole datasets** โดย Rui (Ranger) Fan — <https://github.com/ruirangerfan/stereo_pothole_datasets> (MIT License)

ใน repo นี้เก็บเฉพาะ **ผลลัพธ์ที่ derive แล้ว** ของ 6 เฟรม (`dataset2/24, 26, 30, 33, 40` และ `dataset1/21`):
ภาพถ่ายย่อขนาด, overlay ผลตรวจ, แผนที่มองจากบน และรายงาน — **ไม่เก็บไฟล์ point cloud ดิบ (`.mat`, ~40 MB/เฟรม)**
ถ้าต้องการทำซ้ำ ให้ดาวน์โหลดไฟล์จากต้นทางเอง (ดูวิธีใน README)

กรุณาอ้างอิงงานต้นฉบับเมื่อใช้ข้อมูลนี้ (ตามที่เจ้าของชุดข้อมูลระบุ):

```bibtex
@article{fan2019pothole,
  title={Pothole Detection Based on Disparity Transformation and Road Surface Modeling},
  author={Fan, Rui and Ozgunalp, Umar and Hosking, Brett and Liu, Ming and Pitas, Ioannis},
  journal={IEEE Transactions on Image Processing},
  year={2019},
  publisher={IEEE}
}
@article{fan2018road,
  title={Road surface 3D reconstruction based on dense subpixel disparity map estimation},
  author={Fan, Rui and Ai, Xiao and Dahnoun, Naim},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={6},
  pages={3025--3035},
  year={2018},
  publisher={IEEE}
}
```

### ข้อความ license ของชุดข้อมูล (MIT)

```text
MIT License

Copyright (c) 2021 Rui (Ranger) Fan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
