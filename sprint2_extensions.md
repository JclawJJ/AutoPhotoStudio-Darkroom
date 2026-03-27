# APS Sprint 2 Extensions (from Top-Tier Roundtable)

1. **The Lore-Accurate Database (Grok-3 Hack)**: Don't let the Vision Agent hallucinate backgrounds zero-shot. Create a lookup dictionary mapping detected cosplay IPs to specific prompt anchors (e.g., `Naruto` -> `Hidden Leaf Village`).
2. **ArcFace Consistency Safety Net**: Introduce an `ArcFace` facial similarity check after Phase D. If the clothing modification distorts the user's face (similarity < 0.8), reject and retry. Alternatively, use *AdaIN Style Transfer* on the clothing mask rather than heavy SDXL inpainting.
3. **Latent Couple Multi-Crop**: Automatically output 16:9 (Desktop) and 9:16 (TikTok/Reels) formatted PNGs to encourage viral social sharing.
4. **Offline Convention Mode (Viral Hook)**: A lightweight frontend meant to run on a local GPU at anime conventions. Cosplayers upload via QR code and get a "lore-accurate" cinematic poster in 10 seconds.
5. **GAN Detector-Lite (Claude 3.7 Safety)**: Scan all Phase D outputs for AI artifacts (six fingers, double eyes). Filter bad outputs to a `/review` folder.
- [x] Added Image Format Conversion CLI tooling into APS Pipeline to quickly toggle output/inputs (RAW/JPG/PNG/TIFF) locally inside `--mode convert` feature.
