# Multimodal Fake News Detection using Image Captioning & Three-Way Semantic Fusion

This repository contains the implementation of a multimodal deep learning framework designed to detect fake news by leveraging both **textual and visual signals**. The model identifies semantic inconsistencies between an image and its text caption — a common pattern in misleading social media posts.

The methodology integrates:

* Automated **image captioning** using BLIP
* Text embedding using an ensemble of **BERT, RoBERTa, and DistilBERT**
* Visual embedding using interchangeable backbones (CLIP, Swin Transformer, SigLIP)
* A **three-path fusion mechanism**:

  * Early Fusion
  * Cross-Modal Attention Fusion
  * Late Fusion

---

## Repository Contents

| File                                  | Description                                                                                                 |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `main-model.ipynb`                    | Full implementation of the multimodal fusion model. Supports CLIP/Swin/SigLIP with minor parameter changes. |
| `gen_img_caption.ipynb`               | BLIP-based image caption generator used to enhance textual input before multimodal fusion.                  |

---

## Dataset

This project uses the **Fakeddit Sample Dataset** from HuggingFace:

> **Dataset:** `rtfarchitect/fakeddit_sample`
> **Task Type:** Binary classification (Fake vs Real)

The dataset includes:

* News titles (text modality)
* Image URLs (visual modality)
* Labels (real/fake)

### Preprocessing Steps

* Remove null and duplicate samples
* Caption generation from images using BLIP
* Concatenate: `news_title + generated_caption`
* Stratified train/validation/test split:

  * `70% Train`
  * `15% Validation`
  * `15% Test`

---

## Installation

Ensure Python ≥ 3.9 and install dependencies:

```bash
pip install torch torchvision transformers timm datasets scikit-learn pillow
```

If using GPU support, ensure you install CUDA-compatible PyTorch from the official site.

---

## Running the Project

### Step 1 — Generate Image Captions (Optional but Recommended)

Open and run:

```
gen_img_caption.ipynb
```

This creates an enhanced textual representation using the BLIP model.

---

### Step 2 — Train or Evaluate Multimodal Classifier

Run:

```
main-model.ipynb
```

This notebook performs:

* Text encoding with transformer ensemble
* Visual encoding
* Fusion across three parallel pathways
* Classification (Real vs Fake news)

---

## Switching Visual Models

This framework supports **three visual feature extractors**:

| Visual Model         | Identifier                                 |
| -------------------- | ------------------------------------------ |
| **CLIP**             | `"openai/clip-vit-base-patch32"`           |
| **Swin Transformer** | `"microsoft/swin-base-patch4-window7-224"` |
| **SigLIP**           | `"google/siglip-base-patch16-224"`         |

You only need to update:

```python
CLIP_MODEL = "openai/clip-vit-base-patch32"
```

…and modify the corresponding import line.

> No Structural Changes Needed — Only model/processor lines differ.

---


## Results Summary

Performance varies by visual feature extractor:

| Image Model      | Accuracy   | F1-Score |
| ---------------- | ---------- | -------- |
| Swin Transformer | 0.8806     | 0.88     |
| SigLIP           | 0.8863     | 0.89     |
| **CLIP (Best)**  | **0.8979** | **0.90** |

> CLIP achieved the highest accuracy due to vision-language alignment training — making it more effective for detecting semantic mismatches.

---

## Conclusion

This project demonstrates that multimodal fusion—particularly when enhanced with captioning and vision-language pretrained models—is highly effective for detecting fake news involving mismatched visual and textual content.

The model achieves robust performance and offers a scalable foundation for future misinformation detection systems.

---

## Authors

* **Nishalini K (https://github.com/NishaliniKumar)**
* **Sreenath K A**

B.Tech Artificial Intelligence & Data Science
Shiv Nadar University Chennai

Under the guidance of:
**Dr Balasubramanian P** — IIIT Kottayam


Would you like those additions? 🚀
