from __future__ import annotations

import base64
import io
import json
import time
from typing import Optional

from PIL import Image


SYSTEM_PROMPT = r"""
# Edit Prompt Enhancer
You are a professional edit prompt enhancer. Your task is to generate a direct and specific edit prompt based on the user-provided instruction and the image input conditions.
Please strictly follow the enhancing rules below:
## 1. General Principles
- Keep the enhanced prompt direct and specific.
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.
- All added objects or modifications must align with the logic and style of the edited input image's overall scene.
## 2. Task-Type Handling Rules
### 1. Add, Delete, Replace Tasks
- If the instruction is clear, preserve the original intent and only refine the grammar.
- If the description is vague, supplement with minimal but sufficient details.
### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes.
### 3. Human (ID) Editing Tasks
- Emphasize maintaining the person's core visual consistency.
- For expression changes or beauty changes, they must be natural and subtle.
### 4. Style Conversion or Enhancement Tasks
- Colorization tasks must use: "Restore and colorize the photo."
### 5. Content Filling Tasks
- Inpainting tasks must use: "Perform inpainting on this image. The original caption is: "
- Outpainting tasks must use: "Extend the image beyond its boundaries using outpainting. The original caption is: "
Output a JSON object: {"Rewritten": "..."}
"""


def encode_image_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_rewritten(content: str) -> str:
    text = (content or "").strip().replace("```json", "").replace("```", "").strip()
    payload = json.loads(text)
    return (payload.get("Rewritten") or "").strip().replace("\n", " ")


def rewrite_prompt(
    prompt: str,
    image: Optional[Image.Image],
    *,
    model: str,
    api_key: str | None,
    base_url: str | None,
    max_retries: int = 3,
) -> str:
    prompt = str(prompt or "").strip()
    if not prompt:
        return prompt
    if not api_key:
        return prompt

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    user_content: list[dict[str, object]] = []
    if image is not None:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image_base64_png(image.convert('RGB'))}"},
            }
        )
    user_content.append({"type": "text", "text": f"User Input: {prompt}\n\nRewritten Prompt:"})
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content if image is not None else f"{SYSTEM_PROMPT}\n\nUser Input: {prompt}\n\nRewritten Prompt:"},
    ]

    # MiniMax requires temperature in (0.0, 1.0] — 0.0 is not accepted.
    temperature = 1.0 if ("gpt-5" in model.lower() or model.lower().startswith("minimax")) else 0.0
    last_error = None
    for attempt in range(max_retries):
        try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
            except Exception:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
            rewritten = extract_rewritten(response.choices[0].message.content or "")
            return rewritten or prompt
        except Exception as exc:
            last_error = exc
            time.sleep(0.5 * (2 ** attempt))

    print(f"[PromptRewrite] failed after {max_retries} retries: {last_error}")
    return prompt
