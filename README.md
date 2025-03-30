# Speaker Verification & Speech Enhancement Pipeline

This repository contains the complete pipeline for speaker verification, fine-tuning using LoRA and ArcFace, speech separation using SepFormer, and enhancement in multi-speaker environments. The project is divided into two major tasks and includes both technical reports and executable code.

---

## 📁 Contents

### ✅ Reports
- `Que1_Report.pdf`: End-to-end technical report on speaker verification, fine-tuning, and multi-speaker separation and enhancement.
- `Que2_Report.pdf`: MFCC feature analysis and language classification using Indian language samples.

### 💻 Code Files
- `speech_assignment_task1.py`: Comprehensive pipeline implementing:
  - Pre-trained and fine-tuned speaker verification
  - MFCC-based feature analysis
  - SepFormer-based speaker separation and enhancement
  - Rank-1 speaker identification
  - Complete evaluation metrics: EER, TAR@1%FAR, SDR, SIR, SAR, PESQ

- `task2.ipynb`: (Notebooks assumed to handle MFCC visualizations and language classification using SVM.)

---

## 🧠 Task Descriptions

### 🔹 Task 1: Speaker Verification & Enhancement
- **Dataset:** VoxCeleb1 & VoxCeleb2
- **Model:** HuBERT Large with LoRA + ArcFace for fine-tuning
- **Evaluation Metrics:**
  - Verification: EER, TAR@1%FAR, Accuracy
  - Separation: SDR, SIR, SAR, PESQ
  - Identification: Rank-1 Accuracy

### 🔹 Task 2: Language Identification Using MFCC
- **Dataset:** 10 Indian Languages (3 samples each for Bengali, Gujarati, Hindi)
- **Techniques:**
  - MFCC extraction using `librosa`
  - Visual & statistical MFCC analysis
  - Language classification using SVM

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
