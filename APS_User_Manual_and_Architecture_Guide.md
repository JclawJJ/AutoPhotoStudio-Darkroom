# APS (Agentic Photo Studio) - 商业版产品说明与架构白皮书 (v2026.3)

## 🌌 产品愿景
APS 是一套对标甚至超越千万级融资独角兽（如 Evoto / Pixel Cake）的纯自动化高阶影像后期 SaaS 引擎。
专为商业摄影、影楼批处理及高审美独立创作者设计。彻底告别手工“液化/磨皮”。全本地/去中心化部署，数据零泄露。

## ⚙️ 核心管线架构 (The Trinity Pipeline)
本产品由三座不可撼动的工业级节点构成：

1. **The Sorter (废片品控闸门)**:
   - 核心：内置 104MB MUSIQ / PIQE 深度质量评估模型。
   - 功效：毫秒级自动过滤连拍中的闭眼、糊片、过曝废片（可自定义拦截阈值），绝不让废图浪费后续 GPU 昂贵算力。
2. **The Mapper (亚秒级解耦分割)**:
   - 核心：MediaPipe Face Mesh (0.10.x+) / YOLOv8。
   - 功效：实现真正的全图语义剥离，皮肤、眼白、瞳孔高光、衣物完全分离并生成 Hash 内存缓存。支持 UI 前端亚秒级拖拽响应。
3. **The Forge (算力锻造炉)**:
   - 核心：无头运行的 ComfyUI API 集群 + WAS Node Suite + Impact FaceDetailer。
   - 功效：高低频物理压制，拉普拉斯高频波反向注入，确保任何修容都能 100% 还原索尼 A7M5 的超解析度毛孔细节。

## 🎛️ 核心功能与操作模式

### 模式 A: A7M5 商业高定 (Natural Aesthetic)
- 降噪限死于 `0.25 - 0.35`，结合反向毒药提示词 (`plastic skin, silicone`)，确保质感不发灰，还原东方原生高级感。

### 模式 B: 二次元/网感重构 (Heavy Cosplay / Anime Override)
- 解锁毛孔检测限制，降噪拉升至 `0.55 - 0.65`。
- 允许外部传入特殊风格 Lora 权重，支持重排脸型比例（娃娃脸、陶瓷冷白皮、极致大眼妆）。

### 模式 C: 极速格式转换站 (Format Converter)
- 纯命令行支持：`python aps_pipeline.py --mode convert -i <input> -o <output> --format <jpg/png/tiff>`

## 🔌 第三方 API 接入标准
- APS 微服务已通过 FastAPI 开放：`http://localhost:8000/api/process`
- 直接连通 Next.js (NowHere Inn App) 或任何 Web 前端系统，一次 POST 数据，即刻返回渲染极品图。
