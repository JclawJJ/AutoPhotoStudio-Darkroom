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

with open('/Users/jclaw/.openclaw/workspace/APS_Project/APS_Execution_Plan_Final.md', 'r') as f:
    plan = f.read()

prompt = f"""
Here is our newly updated architecture for 'APS' (Agentic Photo Studio).
It is a 3-tier multimodal pipeline for fully automated, commercial-grade photo retouching built on top of local tools (like ComfyUI) and high-end visual agents.
Recent updates:
- Output for Phase C (Master Filtered) and Phase D (Creative Expansion) has been changed to PNG format, as generative tools like Nano Banana Pro and SDXL are better suited for PNG. Phase B (Retouched Base) retains RAW/DNG.
- Phase D involves contextual reasoning by a Vision Agent: e.g., if it detects Cosplay, it restores the canonical background; if a normal portrait, it keeps character consistency but modifies clothing to match the new background.

Please deeply review this plan. Do not just point out flaws—you must provide actionable engineering solutions, specific feature extensions, and professional creative advice for the APS system.

Document to review:
{plan}
"""

print("\n--- 🧠 Anthropic Claude 3.7 Sonnet (AI Tech Lead / Workflow Master) ---")
print(chat("anthropic/claude-3.7-sonnet", "You are the world's leading AI architectural engineer in 2026. Think deep about AI workflows, memory leakage in ComfyUI, edge cases in Contextual Vision Reasoning.", prompt))

print("\n--- 🧠 OpenAI o3-pro (Creative Director & Photography Architect) ---")
print(chat("openai/o3-pro", "You are an elite creative director who builds world-class AI retouching studios. Focus on aesthetic nuance, the PNG transition, Lora integration, and Phase D magic extensions.", prompt))

print("\n--- 🧠 X.AI Grok 3 (Rebel Systems Auditor) ---")
print(chat("x-ai/grok-3", "You are a sharp, skeptical, and brilliant systems auditor. Find the logical holes in the Cosplay/Character consistency mechanisms, and suggest clever, hacky workarounds or cool extensions to make the pipeline viral.", prompt))

