import sys
import json
import urllib.request
import os

def call_openrouter(model, system, user):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Warning: OPENROUTER_API_KEY not found. Mocking response for safety.", file=sys.stderr)
        return "MOCKED_RESPONSE"
        
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
        "temperature": 0.3
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req) as res:
            response = json.loads(res.read().decode("utf-8"))
            return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling OpenRouter: {e}", file=sys.stderr)
        return f"ERROR: {e}"

def run_roundtable():
    with open('/Users/jclaw/.openclaw/workspace/APS_Project/APS_Execution_Plan_Final.md', 'r') as f:
        plan = f.read()

    user_prompt = f"""
    Here is our proposed architecture for the 'APS' (Agentic Photo Studio) pipeline. It aims to automate commercial retouching (like Evoto) using local AI.
    Pipeline: Sorter (Rank-IQA) -> Mapper (YOLOv8 face + BiSeNet skin segmentation) -> Forge (Headless ComfyUI Impact-Pack/FaceDetailer, denoise 0.25-0.35, dodge/burn, frequency separation).
    Phase B: Retouched Base. High-freq pores kept, neutral gray shadows smoothed.
    Phase C: Master Filtered. Color unification, skin luster (dewy/matte), preset Lora styles.
    Phase D: Creative Expansion. A high-tier Vision Agent analyzes 'C' and dictates prompt logic (removes objects, adapts clothing/cyborg elements if cosplay or normal portrait, and generates sci-fi/cyberpunk backgrounds while keeping character consistency).
    Output: Both RAW (DNG wrapped) and JPG.
    
    Please provide your brutally honest critique and concrete technical suggestions to improve this.
    
    Document to review:
    {plan}
    """

    print("--- 🧠 CLAUDE 3.5 SONNET (AI Architecture Expert) ---")
    claude_sys = "You are a ruthless, top-tier AI Architect. Focus on pipeline efficiency, failure modes, AI Agent logic loops, and the feasibility of saving DNGs and JSON-driven ControlNets."
    res1 = call_openrouter("anthropic/claude-3.5-sonnet", claude_sys, user_prompt)
    print(res1)
    
    print("\n--- 🧠 GPT-4o (Commercial Art Director) ---")
    gpt4_sys = "You are an elite Commercial Art Director and Retoucher. Focus heavily on Eastern aesthetics, skin texture (pores), dodge & burn realism, and whether the vision in Phase C and D will actually look like a $1000/month studio output or cheap AI plastic."
    res2 = call_openrouter("openai/gpt-4o", gpt4_sys, user_prompt)
    print(res2)

if __name__ == "__main__":
    run_roundtable()
