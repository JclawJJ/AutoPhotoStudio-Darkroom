import urllib.request
import json
import os
import sys

def chat(model, system, user):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0.4
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req) as res:
            response = json.loads(res.read().decode("utf-8"))
            return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

plan = """
# APS Sprint 2 - Ultimate Architecture (B-C-D Pipeline)

**Phase B (Retouched Base - RAW/DNG)**: Rank-IQA (Quality Check) -> YOLOv8 Face/Skin Mask -> ComfyUI Impact-Pack FaceDetailer (Locked denoise 0.25-0.35, Frequency Separation dodge/burn).
**Phase C (Master Filtered - PNG)**: Vision Agent (Claude-3.7-Sonnet) analyzes B.jpg -> Outputs JSON -> ComfyUI applies precise Color Match & targeted aesthetic Lora.
**Phase D (Creative Expansion - PNG)**: 
  1. *Archetype DB Recognition*: Is it Cosplay (e.g., Naruto) or Standard?
  2. *Hack Tool - Danbooru Lore DB*: Map cosplay tags to precise canonical background prompts.
  3. *Hack Tool - ArcFace Similarity Check*: Ensure face consistency >= 0.85 after replacing background (Outpaint) or altering clothing (AdaIN Style Transfer).
  4. *Hack Tool - Latent Couple*: Output dual-crop variants (16:9 PC, 9:16 TikTok).

Are there specific GitHub open-source repositories we can leverage for the 'Danbooru Lore DB', 'ArcFace Consistency', and 'AdaIN Style Transfer' to accelerate development? Approve this final blueprint.
"""

print("\n--- 🧠 Anthropic Claude 3.7 Sonnet (AI Tech Lead) ---")
print(chat("anthropic/claude-3.7-sonnet", "You are the ultimate Engineering Lead. Provide exact GitHub repo links or standard HuggingFace models for ArcFace, AdaIN, and anime tag databases. Give your final blessing to this robust architecture.", plan))

print("\n--- 🧠 X.AI Grok 3 (Rebel Systems Auditor) ---")
print(chat("x-ai/grok-3", "You are a hacker auditor. Bless this plan and point out exact open-source tools/repos we can stitch together to make the D-Phase (ArcFace + AdaIN + Danbooru DB) a reality without building from scratch.", plan))
