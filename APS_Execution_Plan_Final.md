# APS (Agentic Photo Studio) - 全量本地化自动修图开发蓝图

> **Status:** Approved via Midnight Roundtable Protocol (2026-03-25)
> **Goal:** 针对 Sony A7M5 直出照片，实现一套完全无需人工干预、对标 Evoto/像素蛋糕 商业级品质的自动化选片与精修工作流。

---

## 1. 审美契约 (The Aesthetic Baseline)
经过对行业顶配修图心法的拆解，APS 管线必须在底层锁死以下修图逻辑，彻底弃用低级 AI 的“硅胶涂抹”：
*   **频率分离守护 (Frequency Separation)**：高频层绝对保留原生毛孔与皮脂颗粒感；低频层通过色调均衡解决痘印、暗沉与肤色不均。
*   **中性灰重构 (Dodge & Burn)**：针对眼袋、法令纹与颧骨阴影，使用非破坏性填光算法，而非粗暴的全局液化拉伸。
*   **东亚自然骨相**：“修而无痕”，极其克制的液化瘦身，重点在于面部立体轮廓（高光/阴影）的强化。

## 2. 核心架构与技术缝合选型 (The Architecture)
经过旗舰模型圆桌与社区开源轮子 (GitHub/Reddit) 的三轮交叉毒打，最终确立一套 **“完全解耦 UI 的纯后台脚本流”**：

### Module A: 智能分拣器 (The Sorter)
*   **职责**：淘汰对焦失败、曝光严重不足/过度的废片。
*   **组件**：选用开源的 `Rank-IQA` 或 `PIQE` 算法模块。仅需几行 Python 代码即可批量评估图像锐度与信噪比，卡死 0.7 阈值，直出优选池。

### Module B: 皮肤分割与特征锚定 (The Mapper)
*   **职责**：精准剥离“需保护区”（眼睛、毛发、背景）与“重绘区”（面部与躯干皮肤）。
*   **组件**：结合 `ultralytics/yolov8m-face` 与轻量级 `BiSeNet-V2` 皮肤语义分割。为后续重绘提供像素级精准的 mask 蒙版。

### Module C: 核心重构引擎 (The Forge - ComfyUI CLI)
*   **核心引擎**：舍弃笨重的客户端，直接通过 **`ComfyUI CLI` (Headless 模式)** 调用写死的工作流 JSON。
*   **节点配置 (The Secret Sauce)**：
    *   搭载最强社区轮子 **`ComfyUI-Impact-Pack` (FaceDetailer)**。
    *   **避坑参数硬编码**：为解决 Reddit 哀嚎的黑白噪点和假死问题，将 `guide_size` 锁死在最佳容差值，`denoise` 严格限制在 0.25~0.35 之间（仅修饰不改变结构）。
    *   **大师滤镜层**：工作流内直接挂载风格化 Lora，预注入 System Prompt 提示词：`"smooth skin, natural tone, asian aesthetic, macro photography, highly detailed"`。

## 3. 分期开发路线图 (Execution Sprints)

*   **Sprint 1: 基础设施与黑盒验证**
    *   使用 Python 编写外围调度脚本 `aps_pipeline.py`。
    *   跑通无头环境下的图像读取与质量打分 (Rank-IQA) 阶段。
*   **Sprint 2: ComfyUI 引擎点火**
    *   在后台部署纯净版 ComfyUI 及相关节点（Impact Pack）。
    *   编写并在脚本中触发 `facial_retouch_master.json` 工作流，验证 FaceDetailer 的参数边界。
*   **Sprint 3: 连点成线与压测**
    *   将 50 张 Sony A7M5 测试集扔进脚本，测试 Mac 本地显存切分与批量生存能力。输出最终对比报告 (B/A Preview)。

---
*编者按：此模块由于全链路可通过 JSON / Python 脚本化触发，未来将完美平移、融入 RedIris (NowHere Inn) 的全局 AIGC 微服务中，作为其高清原画库与 Lora 清洗工厂的底层驱动。*
## Sprint 2: APS Multimodal Studio Evolution (The Master's Trinity)
**Master Directive (2026-03-25):** Add a high-tier multi-modal Agent loop into the pipeline to drive iterative aesthetic evolution (B → C → D). Output all stages in both RAW/DNG and JPG formats.

1. **Phase B (Retouched Base):** 
   - The output of Sprint 1 (Frequency Separation & Netural Gray Dodge & Burn).
   - High-freq pores retained.
   - Saves: `B_Retouched.jpg` & `B_Retouched.dng` (or TIF wrapping).
2. **Phase C (Master Filtered):**
   - **Agent Read:** Feeds compressed `B_Retouched.jpg` to a high-tier Vision Agent.
   - **Agent Analysis:** Agent suggests specific Master-level color grading, lighting adjustment, and Lora styles (e.g., Cinematic Teal/Orange, Cyberpunk Neon).
   - **Execution:** APS uses ComfyUI (Color Match / Lora injection) to apply the requested styles. 
   - Saves: `C_MasterFilter.png` with JSON Prompt Metadata.
3. **Phase D (Creative Expansion):**
   - **Agent Read:** Feeds compressed `C_MasterFilter.jpg` back to the Vision Agent. 
   - **Agent Analysis:** Agent detects distracting background elements or suggests total sci-fi/cinematic background overhauls. 
   - **Execution:** Calls tools like `Nano Banana Pro` (Gemini 3 Pro Image) or ComfyUI SDXL Inpaint/Outpaint to replace the background.
   - Saves: `D_Expansion.png` with Action Ledger JSON.

### Phase D (Creative Expansion) - Deep Logic & Reasoning (Master Directive 17:23)
- **Agent Reasoning:** The Vision Agent analyzing Image C must employ a strict, contextual reasoning tree:
  - Is the subject a specific archetype/Cosplay? If yes, precisely repair and complete the canonical fictional background/lighting.
  - Is it a standard portrait? Then extract the subject, maintain high personal consistency, but modify clothing/texture contextually to match the newly generated sci-fi/fantasy/cyberpunk background.
- **Execution:** Zero-shot random generation is prohibited. The prompt fed to the generative engine must be grounded in the context understood by the Vision Agent.

## Sprint 2 Open-Source Integrations (Blessed by Roundtable)
*   **DeepDanbooru (Archetype Recognition):** `https://github.com/KichangKim/DeepDanbooru` for extracting character tags and pulling canonical background text prompts.
*   **InsightFace / ArcFace (Consistency Check):** `https://github.com/deepinsight/insightface` (`glintr100.onnx`). Bounding box from YOLOv8 is fed here. Threshold = 0.85 cosine similarity.
*   **AdaIN Style Transfer:** `https://github.com/naoto0804/pytorch-AdaIN`. Used for modifying clothing textures to match the sci-fi/cyberpunk backgrounds without messing up the geometry.
*   **Latent Couple:** `https://github.com/opparco/comfyui-latent-couple` for automated 16:9 and 9:16 dual-crop renders.
