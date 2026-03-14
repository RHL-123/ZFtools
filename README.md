# ZFtools

手机刷机工具集合，包含以下工具：

## 工具列表

1. MiFlash (2022.5.7.0)
   - 小米官方刷机工具
   - 支持线刷/fastboot模式

2. Odin3 (3.13.3)
   - 三星官方刷机工具
   - 支持线刷模式

3. SP Flash Tool
   - 联发科官方刷机工具
   - 版本：v5.1812.00, v5.2044.00, v5.2216.00, v6.2228

4. 其他工具
   - 小米ADB刷机环境
   - 搞机工具箱
   - 网盘资料下载神器
   - 各品牌手机驱动

## 目录结构

```
ZFtools/
├── data/
│   └── tool/
│       ├── MiFlash_2022.5.7.0/
│       ├── Odin3 3.13.3/
│       └── SP_Flash_Tool_v*/
├── driver/
│   ├── MTK开机改串号驱动/
│   ├── 华为强开生产模式/
│   └── 其他品牌驱动/
└── README.md
```

---

## 目前最好用的人脸识别模型

以下是目前（2024–2025 年）综合评价最高、最常用的人脸识别模型，涵盖检测、对齐与识别三个阶段。

### 🏆 综合推荐：InsightFace（ArcFace）

| 项目 | 说明 |
|------|------|
| **项目地址** | https://github.com/deepinsight/insightface |
| **核心算法** | ArcFace / AdaFace / CosFace |
| **特点** | 精度高、开源免费、模型丰富、支持 Python/ONNX/TensorRT |
| **适用场景** | 门禁、安防、手机解锁、人脸搜索 |
| **推荐模型** | `buffalo_l`（高精度）/ `buffalo_s`（轻量） |

**安装与快速使用（Python）：**

```bash
pip install insightface onnxruntime
```

```python
import insightface
from insightface.app import FaceAnalysis
import cv2

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

img = cv2.imread('test.jpg')
faces = app.get(img)
for face in faces:
    print('人脸框:', face.bbox)
    print('人脸特征向量维度:', face.embedding.shape)  # (512,)
```

---

### 主流模型横向对比

| 模型 | 识别精度 | 速度 | 易用性 | 开源 | 适用场景 |
|------|---------|------|--------|------|---------|
| **InsightFace (ArcFace)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | 通用、安防、门禁 |
| **DeepFace** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | 研究、快速验证 |
| **FaceNet** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | 人脸验证、1:1 比对 |
| **AdaFace** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | 低质量图像、遮挡场景 |
| **RetinaFace** | 检测专用⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | 人脸检测与关键点定位 |
| **百度 PaddleFace** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | 国内部署、移动端 |

---

### 各模型简介

#### 1. InsightFace / ArcFace
- **地址**：https://github.com/deepinsight/insightface
- 目前学术界和工业界最广泛使用的人脸识别框架，LFW 准确率 **99.83%**。
- 提供预训练模型 `buffalo_l`、`buffalo_sc` 等，支持 ONNX 导出，可在 CPU/GPU 上运行。
- 内置 RetinaFace 检测 + ArcFace 识别的完整 pipeline。

#### 2. DeepFace
- **地址**：https://github.com/serengil/deepface
- 轻量封装库，底层可切换多种后端（ArcFace、Facenet、VGG-Face、DeepID 等）。
- 一行代码完成人脸验证，适合快速原型开发。

```bash
pip install deepface
```

```python
from deepface import DeepFace
result = DeepFace.verify('img1.jpg', 'img2.jpg', model_name='ArcFace')
print(result['verified'])  # True / False
```

#### 3. FaceNet
- **地址**：https://github.com/davidsandberg/facenet（原版 TF1）/ https://github.com/timesler/facenet-pytorch（PyTorch）
- Google 出品，Triplet Loss 训练，512 维特征向量，适合 1:1 人脸比对。

#### 4. AdaFace
- **地址**：https://github.com/mk-minchul/AdaFace
- 针对低质量、遮挡、侧脸场景优化，在 IJB-C 等困难数据集上超越 ArcFace。

#### 5. RetinaFace（人脸检测）
- **地址**：https://github.com/deepinsight/insightface/tree/master/detection/retinaface
- 目前精度最高的人脸检测模型之一，同时输出人脸框和 5 个关键点（眼、鼻、口角）。

#### 6. PaddleFace / PP-YOLOE-R（百度飞桨）
- **地址**：https://github.com/PaddlePaddle/PaddleClas
- 国内推荐，移动端部署友好，Lite 版本适合安卓/iOS/嵌入式设备。

---

### 选型建议

| 需求 | 推荐方案 |
|------|---------|
| 服务器端高精度识别 | InsightFace `buffalo_l` + RetinaFace |
| 快速原型 / 研究验证 | DeepFace（ArcFace 后端） |
| 移动端 / 边缘设备 | InsightFace `buffalo_s` 或 PaddleFace Lite |
| 遮挡 / 低质量照片 | AdaFace |
| 纯人脸检测 | RetinaFace 或 MTCNN |