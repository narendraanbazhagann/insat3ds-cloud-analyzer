# ============================================================
# MOSDAC INSAT-3DS TEMPORAL CLOUD ANALYZER
# Qwen 2.5 VL 3B + Gradio
# ============================================================

!pip -q install -U gradio transformers accelerate qwen-vl-utils requests pillow


# ============================================================
# IMPORTS
# ============================================================

import os
import gc
import torch
import requests
import gradio as gr

from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
from io import BytesIO

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

print("Loading Qwen 3B...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

print("Qwen 3B loaded successfully!")

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# MOSDAC SETTINGS
# ============================================================

BASE_URL = (
    "https://mosdac.gov.in/look/3S_IMG/preview/"
)

SAVE_DIR = Path("/content/mosdac_nearest")
SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MOSDAC URL
# ============================================================

def make_mosdac_url(timestamp):

    day = timestamp.strftime("%d")
    month = timestamp.strftime("%b").upper()
    year = timestamp.strftime("%Y")
    hhmm = timestamp.strftime("%H%M")

    filename = (
        f"3SIMG_{day}{month}{year}_{hhmm}"
        f"_L1C_SGP_IR1_V01R00.jpg"
    )

    return (
        f"{BASE_URL}"
        f"{year}/{day}{month}/"
        f"{filename}"
    )


# ============================================================
# GET MOSDAC IMAGE
# ============================================================

def get_mosdac_image(timestamp):

    url = make_mosdac_url(timestamp)

    try:

        response = requests.get(
            url,
            timeout=12
        )

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if (
            response.status_code == 200
            and "image/jpeg" in content_type
        ):

            image = Image.open(
                BytesIO(response.content)
            ).convert("RGB")

            return image, url

    except Exception:
        pass

    return None, None


# ============================================================
# FIND NEAREST IMAGE
# ============================================================

def find_nearest_image(
    target,
    search_window=60
):

    # Search:
    # exact
    # -10, +10
    # -20, +20
    # etc.

    offsets = [0]

    for minutes in range(
        10,
        search_window + 1,
        10
    ):

        offsets.append(-minutes)
        offsets.append(minutes)

    for offset in offsets:

        candidate = (
            target +
            timedelta(minutes=offset)
        )

        image, url = get_mosdac_image(
            candidate
        )

        if image is not None:

            return {
                "target_time": target,
                "actual_time": candidate,
                "difference": abs(offset),
                "image": image,
                "url": url
            }

    return None


# ============================================================
# TEMPORAL PROMPT
# ============================================================

def build_prompt(
    results,
    start_time,
    end_time
):

    timeline = []

    for i, item in enumerate(
        results,
        start=1
    ):

        target = item[
            "target_time"
        ].strftime(
            "%Y-%m-%d %H:%M"
        )

        actual = item[
            "actual_time"
        ].strftime(
            "%Y-%m-%d %H:%M"
        )

        diff = item[
            "difference"
        ]

        timeline.append(
            f"Image {i}: "
            f"Target={target}, "
            f"Actual={actual}, "
            f"Difference={diff} minutes"
        )

    timeline_text = "\n".join(
        timeline
    )

    return f"""
You are analyzing chronological MOSDAC INSAT-3DS
thermal infrared satellite images.

These images belong to ONE continuous time period.

TIME PERIOD:

{start_time.strftime("%Y-%m-%d %H:%M")}
to
{end_time.strftime("%Y-%m-%d %H:%M")}

ACTUAL IMAGE TIMELINE:

{timeline_text}

Analyze all images together as a SINGLE temporal sequence.

Do NOT describe each image independently.

The goal is to explain what happened between the
beginning and end of the time period.

Focus on:

- cloud systems appearing
- cloud systems disappearing
- cloud expansion
- cloud contraction
- apparent movement
- movement direction when visually supported
- changes in cloud-top appearance
- strengthening or weakening of cloud activity
- persistent cloud systems
- changes over India
- changes over the Bay of Bengal

IMPORTANT RULES:

1. Only make observations supported by the images.
2. Do not invent coordinates.
3. Do not invent weather events.
4. Do not automatically call something a cyclone,
   depression or tropical storm.
5. Thermal infrared brightness alone does not prove rainfall.
6. If movement cannot be determined, say:
   "Movement is unclear from the available images."
7. Compare the images chronologically.
8. Mention the actual observation period.
9. Keep the answer concise.
10. Focus on CHANGE THROUGH TIME.

Return ONLY the following structure:

======================================================================
QWEN TEMPORAL ANALYSIS
======================================================================

### TIME PERIOD

State the actual observation period.

### OVERALL CHANGE

Explain in 2-4 sentences what changed from the
earliest image to the latest image.

### CLOUD DEVELOPMENT

Explain:

- expansion
- contraction
- formation
- dissipation
- persistence

Focus on the major cloud systems.

### CLOUD MOVEMENT

Explain whether the major cloud systems appear
to move.

If movement is visible, state the approximate
direction.

If movement cannot be determined, say:

"Movement is unclear from the available images."

### INDIA

Explain the important cloud changes over India
during the observation period.

### BAY OF BENGAL

Explain the important cloud changes over the
Bay of Bengal during the observation period.

### CONVECTION

Explain whether cloud-top activity appears to
strengthen, weaken or remain relatively stable.

Do NOT directly equate this with rainfall.

### IMPORTANT CHANGE

Finish with 1-2 sentences describing the most
important change during the entire period.

Do not create tables.
Do not create a formal weather report.
Do not repeat the same observation unnecessarily.
"""


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_mosdac(
    start_text,
    end_text,
    number_images,
    search_window,
    progress=gr.Progress()
):

    # --------------------------------------------------------
    # Parse dates
    # --------------------------------------------------------

    try:

        start = datetime.strptime(
            start_text.strip(),
            "%Y-%m-%d %H:%M"
        )

        end = datetime.strptime(
            end_text.strip(),
            "%Y-%m-%d %H:%M"
        )

    except Exception:

        return (
            "❌ Invalid date/time format.\n\n"
            "Use:\n"
            "`YYYY-MM-DD HH:MM`"
        )

    if end <= start:

        return (
            "❌ End time must be after "
            "the start time."
        )

    number_images = int(
        number_images
    )

    search_window = int(
        search_window
    )

    if number_images < 2:

        return (
            "❌ Number of images must "
            "be at least 2."
        )

    # --------------------------------------------------------
    # Generate timestamps
    # --------------------------------------------------------

    total_seconds = (
        end - start
    ).total_seconds()

    interval = (
        total_seconds /
        (number_images - 1)
    )

    targets = [

        start +
        timedelta(
            seconds=interval * i
        )

        for i in range(
            number_images
        )
    ]

    # --------------------------------------------------------
    # Find nearest MOSDAC images
    # --------------------------------------------------------

    results = []

    for i, target in enumerate(
        targets
    ):

        progress(
            (i + 1) / number_images,
            desc=f"Finding MOSDAC image {i+1}/{number_images}"
        )

        result = find_nearest_image(
            target,
            search_window
        )

        if result is None:
            continue

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        filename = (
            f"{i+1:02d}_"
            f"{result['actual_time'].strftime('%Y%m%d_%H%M')}"
            ".jpg"
        )

        filepath = (
            SAVE_DIR /
            filename
        )

        result[
            "image"
        ].save(
            filepath,
            quality=95
        )

        result[
            "path"
        ] = str(filepath)

        results.append(
            result
        )

    # --------------------------------------------------------
    # No images
    # --------------------------------------------------------

    if len(results) == 0:

        return (
            "❌ No MOSDAC images were found "
            "within the selected search window."
        )

    # --------------------------------------------------------
    # Resize images for T4
    # --------------------------------------------------------

    images = []

    for item in results:

        img = item[
            "image"
        ].copy()

        img.thumbnail(
            (768, 768),
            Image.Resampling.LANCZOS
        )

        images.append(
            img
        )

    # --------------------------------------------------------
    # Build Qwen prompt
    # --------------------------------------------------------

    prompt = build_prompt(
        results,
        start,
        end
    )

    # --------------------------------------------------------
    # Build multimodal message
    # --------------------------------------------------------

    content = []

    for image in images:

        content.append({
            "type": "image",
            "image": image
        })

    content.append({
        "type": "text",
        "text": prompt
    })

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]

    # --------------------------------------------------------
    # Prepare Qwen input
    # --------------------------------------------------------

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt"
    )

    # --------------------------------------------------------
    # Move tensors to GPU
    # --------------------------------------------------------

    inputs = {
        key: value.to(
            model.device
        )
        if hasattr(value, "to")
        else value

        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    progress(
        0.95,
        desc="Qwen analyzing cloud evolution..."
    )

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False
        )

    # --------------------------------------------------------
    # Remove input tokens
    # --------------------------------------------------------

    input_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )

    generated_ids = (
        output_ids[
            :,
            input_length:
        ]
    )

    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del inputs
    del output_ids
    del generated_ids

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Return ONLY analysis
    # --------------------------------------------------------

    return answer


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="MOSDAC Cloud Timeline Analyzer"
) as demo:

    gr.Markdown(
        """
# 🛰️ MOSDAC Cloud Timeline Analyzer

Analyze **INSAT-3DS thermal infrared imagery over time**
using **Qwen 2.5 VL 3B**.

Select a start time and end time. The system will
automatically find the nearest available MOSDAC images.
"""
    )

    with gr.Row():

        start_input = gr.Textbox(
            label="Start Date & Time",
            value="2026-09-24 00:00",
            placeholder="YYYY-MM-DD HH:MM"
        )

        end_input = gr.Textbox(
            label="End Date & Time",
            value="2026-09-25 00:00",
            placeholder="YYYY-MM-DD HH:MM"
        )

    with gr.Row():

        number_input = gr.Number(
            label="Number of Images",
            value=5,
            minimum=2,
            maximum=8,
            precision=0
        )

        search_input = gr.Number(
            label="Nearest Image Search Window (minutes)",
            value=60,
            minimum=10,
            maximum=120,
            precision=0
        )

    analyze_button = gr.Button(
        "🚀 Analyze Cloud Evolution",
        variant="primary"
    )

    output = gr.Markdown(
        label="Temporal Analysis"
    )

    analyze_button.click(
        fn=analyze_mosdac,
        inputs=[
            start_input,
            end_input,
            number_input,
            search_input
        ],
        outputs=output
    )


# ============================================================
# LAUNCH
# ============================================================

demo.launch(
    share=True,
    debug=False
)
