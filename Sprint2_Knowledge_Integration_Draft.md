# APS Sprint 2 - Universal Knowledge Integration Architecture (Commercial Final)

## 核心目标
将通用级商业摄影视觉准则与生成式AI工作流进行物理级无缝缝合。确立 APS Phase C（自动色调与滤镜）与 Phase D（通用场景延展与重构）的底层判断逻辑，确保最终输出绝对符合专业影棚与商业人像的标准，剥离任何生图模型的低级“AI 塑料感”。

## 泛用级双轴缝合管线 (The Dual-Axis Integration)

### Axis 1: 光影与质感绝对锚定 (Phase C 通用光学校准)
- **挂载外部知识域**: `ai-product-photography` (商业布光参数与质感锐化白皮书)
- **职能**: 锁死人物皮肤细节与全域结构光影底限。提取其针对“表面微反光、漫反射、硬光柔化”的绝对物理算法描述。
- **Agent Action**: Vision Agent 审视中性灰修图（Image B）后，向 ComfyUI (Color Match / Dodge & Burn 节点) 传递色彩与打光推荐时，必须附带此规则设定的硬限制指令：任何全图滤镜与高光重构，绝对不允许出现“扁平、发灰、塑料泛滥”等反物理伪像。必须保留 Sony A7M5 直出的高级商拍立体纵深。

### Axis 2: 焦段与景深物理透视解构 (Phase D 场景重构)
- **挂载外部知识域**: `google-imagen-3-portrait-photography` (纯光学构图与透视指南)
- **职能**: 保证任何换背景或衣物的 AI 生成内容，其光学属性必须 100% 服从于原图。
- **Agent Action**: 当 Vision Agent (如使用 Gemini 3 Pro Vision) 在 Phase D 决定替换杂乱背景或拓展通用风格时，必须率先根据此规则分析原图的焦段结构（如明确它是 85mm f/1.4 虚化还是 35mm 广角硬光）。传递给 ComfyUI 的替换提示词必须强制包含等效的物理光学描述，从而解决前景与后景割裂的抠图感。

## 圆桌会审的优化靶点 (Red Team Audit)
1. **指令长度癌变 (CLIP 77 Token Limit)**: 强加商业参数极度耗费词元。如果这套物理描述极长，将超出 77 Token 截断限制。
   - 制衡方案：要求 CC 开发一套“标签组 (Tag Arrays) 转换字典”，使用 `(keyword:1.5)` 高权重形式替代繁琐的长句介词描述。
2. **Vision Agent 的过激重构冲动**: 当引入高阶摄影模型指导后，大模型极易“过度思考”从而对完好的人像本身进行面部破坏。
   - 制衡方案：命令 CC 在 Python 脚本与 System Prompt 顶层增加遮罩优先级硬锁 (Mask Priority Hardlock)，重构算法严禁触及原脸轮廓。## Phase E: Custom Intensity Overrides (Cosplay/Anime Style Support)
- **User Request**: The system must support extreme visual transformations (comparable to TikTok/Xiaohongshu cosplay heavy edits).
- **Implementation**: Expose an `intensity_override` / `style_preset` slider in the UI and API.
  - When standard (A7M5 Natural), denoise is locked at `0.25-0.35` to strictly prevent plastic looks and preserve pores.
  - When "Cosplay/Anime/Heavy Aesthetic" is selected, bypass the MUSIQ/PIQE skin variance checks. Allow denoise to scale up to `0.55-0.65`, introduce Lora weights for stylized 2D/2.5D rendering, and enable heavy facial restructuring (FaceDetailer guide_size scaling). 
  - **Conclusion**: Yes, entirely possible by dynamically relaxing our Phase B constraints and routing it through a stylized ComfyUI sub-workflow.
