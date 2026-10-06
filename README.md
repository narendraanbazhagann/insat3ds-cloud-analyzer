# MOSDAC INSAT-3DS Temporal Cloud Analyzer

An AI-powered meteorological analysis system that retrieves **INSAT-3DS infrared satellite imagery** from India's official **MOSDAC** repository and uses **Qwen2.5-VL-3B-Instruct** to analyze cloud dynamics across multiple points in time.

The system is designed to identify and describe **cloud movement, expansion, contraction, convection trends, and temporal changes** over regions including India, the Indian Ocean, and the Bay of Bengal.

## 🚀 Key Features

- 🛰️ **MOSDAC Satellite Data Retrieval**
  - Retrieves INSAT-3DS satellite imagery from MOSDAC.
  - Searches for the closest available image when an exact timestamp is unavailable.

- ⏱️ **Temporal Image Sampling**
  - Accepts a start time, end time, and number of images.
  - Automatically generates uniformly spaced timestamps.
  - Searches temporal offsets around each target timestamp.

- 🔎 **Nearest-Image Retrieval**
  - Searches timestamps using an offset pattern:
  
  `0 → -10 → +10 → -20 → +20 → ...`

  - Supports a configurable search window such as ±60 minutes.

- 🖼️ **Memory-Optimized Processing**
  - Satellite imagery is resized to `768 × 768`.
  - Uses Lanczos resampling to preserve image quality.
  - Designed to operate within the memory limitations of a Tesla T4 GPU.

- 🤖 **Multimodal Vision-Language Analysis**
  - Uses **Qwen2.5-VL-3B-Instruct**.
  - Multiple satellite images are provided together as a chronological sequence.
  - The model is prompted to analyze changes across time rather than treating each image independently.

- 🌩️ **Cloud Dynamics Analysis**
  - Cloud movement direction
  - Cloud system expansion
  - Cloud contraction
  - Convective development
  - Dissipation
  - Temporal cloud-system evolution

- 🌐 **Interactive Gradio Interface**
  - Browser-based interface.
  - User-selectable time range.
  - Configurable number of satellite images.
  - Automated retrieval and AI analysis.

---

## 🏗️ System Architecture

```text
                User
                 │
                 ▼
        ┌─────────────────┐
        │  Gradio Web UI  │
        └────────┬────────┘
                 │
        Start / End / N Images
                 │
                 ▼
       ┌─────────────────────┐
       │ Temporal Scheduler  │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ MOSDAC Retrieval    │
       │ Nearest Timestamp   │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Image Preprocessing │
       │ Resize → 768×768    │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Chronological Image │
       │ Sequence            │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Qwen2.5-VL-3B       │
       │ Multimodal Analysis │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Cloud Dynamics      │
       │ Analysis            │
       └─────────────────────┘
```

## 🔄 Workflow

### 1. Define Analysis Window

The user provides:

- Start date and time
- End date and time
- Number of satellite images

For example:

```text
Start: 2026-10-01 06:00
End:   2026-10-01 18:00
Images: 5
```

The system calculates five uniformly distributed target timestamps.

### 2. Find Available Satellite Images

Satellite imagery may not exist at the exact requested minute.

Therefore, the system performs a nearest-neighbor temporal search:

```text
Target timestamp
      │
      ├── 0 min
      ├── -10 min
      ├── +10 min
      ├── -20 min
      ├── +20 min
      ├── ...
      └── ±60 min
```

The closest available MOSDAC image is selected.

### 3. Image Preprocessing

Retrieved images are resized to:

```text
768 × 768
```

Lanczos resampling is used to maintain image quality while reducing GPU memory consumption.

### 4. Temporal Sequence Construction

Images are ordered chronologically:

```text
Image 1 → T1
Image 2 → T2
Image 3 → T3
Image 4 → T4
Image 5 → T5
```

All images are passed to the vision-language model within a single conversational context.

### 5. AI Analysis

Qwen2.5-VL-3B-Instruct is instructed to focus on **temporal changes** rather than producing independent descriptions.

The analysis considers:

- Movement of cloud systems
- Expansion and contraction
- Formation and dissipation
- Convective development
- Directional movement
- Changes in cloud density and structure

### 6. Results

The system generates a natural-language meteorological interpretation of the observed cloud evolution.

---

## 🧠 Why Qwen2.5-VL?

Traditional image captioning would analyze each satellite image independently:

```text
Image 1 → Description
Image 2 → Description
Image 3 → Description
```

This project instead provides the images as a temporal sequence:

```text
Image 1 ─┐
Image 2  │
Image 3  ├──► Qwen2.5-VL ──► Temporal Analysis
Image 4  │
Image 5 ─┘
```

This allows the model to reason about **how the observed cloud systems change over time**.

---

## ⚡ GPU Optimization

The project is designed with limited GPU memory in mind.

The main optimizations include:

| Optimization | Purpose |
|---|---|
| `768 × 768` images | Reduce visual-token memory |
| Limited number of images | Control sequence size |
| `max_new_tokens=500` | Limit generation memory |
| `gc.collect()` | Release Python objects |
| `torch.cuda.empty_cache()` | Release unused CUDA memory |
| `device_map="auto"` | Automatically place model components |

These optimizations are particularly useful when running on a **Tesla T4 with 16 GB VRAM**.

---

## 🛠️ Tech Stack

### AI / Machine Learning
- Qwen2.5-VL-3B-Instruct
- Hugging Face Transformers
- PyTorch

### Satellite / Meteorological Data
- MOSDAC
- INSAT-3DS

### Image Processing
- Pillow
- Lanczos resampling

### Web Interface
- Gradio

### Programming
- Python

### Hardware
- NVIDIA Tesla T4 GPU

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/<YOUR_USERNAME>/mosdac-insat3ds-temporal-cloud-analyzer.git

cd mosdac-insat3ds-temporal-cloud-analyzer
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

---

## 📋 Example Input

```text
Start Date: 2026-10-01 06:00
End Date:   2026-10-01 18:00
Images:     5
```

The system retrieves the closest available satellite images and sends them to the multimodal model for temporal analysis.

---

## 📊 Example Output

The model can produce observations such as:

```text
The cloud system shows gradual eastward movement during
the observation period. Convective activity increases in
the later frames, with the cloud cluster becoming more
organized and spatially extensive.
```

> Note: The model output is an AI-assisted interpretation of satellite imagery and should not be treated as an official meteorological forecast.

---

## 🔬 Research / Engineering Highlights

This project combines several components into a single pipeline:

```text
Satellite Data Retrieval
        +
Temporal Search
        +
Image Optimization
        +
Multimodal Vision
        +
Temporal Reasoning
        +
Interactive Deployment
```

The main engineering challenge is balancing **temporal context, image resolution, model inference, and GPU memory** when processing multiple satellite images simultaneously.

---

## 🗺️ Future Improvements

- [ ] Add automatic weather-system tracking
- [ ] Add cloud-motion vector estimation
- [ ] Generate animated satellite sequences
- [ ] Add precipitation/radar data integration
- [ ] Add cyclone detection and tracking
- [ ] Add geographic region selection
- [ ] Add historical temporal comparisons
- [ ] Store analysis results in a database
- [ ] Add automated meteorological report generation
- [ ] Deploy using Hugging Face Spaces or a GPU cloud platform

---

## ⚠️ Limitations

- Satellite image availability depends on MOSDAC.
- The nearest available image may not exactly match the requested timestamp.
- VLM-generated observations are interpretations and may contain errors.
- Processing multiple high-resolution images requires significant GPU memory.
- The system is intended for research and analysis, not operational weather forecasting.

---

## 📄 License

This project is intended for research and educational purposes.

Add the appropriate license before publishing the repository.

---

## 👨‍💻 Author

**Narendra A**

Artificial Intelligence & Machine Learning

Research & Development Project

---

⭐ If you find this project useful, consider giving the repository a star.
