# 🌿 Green MLOps: Bio-Inspired Energy-Aware Inference

**Green MLOps** is a closed-loop inference framework that prioritizes energy efficiency without sacrificing significant accuracy. Inspired by **biophysics (protein folding gradients)**, this project implements a dynamic controller that filters and routes inference requests based on their estimated "metabolic cost" (uncertainty & system congestion).

![Bio-Inspired Controller](Folding.png)

## 🚀 Key Features
- **Bio-Inspired Controller**: A dynamic threshold mechanism ($\tau(t)$) that decides when to execute or skip inference, effectively acting as an "Early Exit".
- **Dual-Path Serving**:
  - **Path A**: Low-latency local inference via **FastAPI** + **ONNX Runtime (ORT)**.
  - **Path B**: High-throughput batched inference via **NVIDIA Triton Inference Server**.
- **Real-Time Sustainability Metrics**: Tracks Carbon ($gCO_2eq$) and Energy ($kWh$) per request using **CodeCarbon**.
- **Validated on A100**: Achieved **42% latency reduction** with <0.5% accuracy loss on DistilBERT/ResNet benchmarks.
- 
![Dashboard](dashboard.png)
Real-world deployment: SmartDiag Radiology Dashboard powered
by our Green MLOps stack. The controller manages multimodal inferences
for tumor detection (red bounding box), balancing A100 energy consumption
against diagnostic latency requirements.
- 

## 📂 Repository Structure
- `src/green_mlops.py`: The main Python script containing the full pipeline (Setup, Model Export, Bio-Controller Logic, Benchmark).
- `GreenMLOps_IEEE_Paper.pdf`: The associated research paper detailing the theory and results.
- `images/`: Visual assets (Architecture diagram, Bio-inspired landscapes).

## 🛠️ Usage

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Controller Benchmark
Execute the main script. It will automatically export models to ONNX (if not present) and run the comparison between Standard and Bio-Inspired inference.

```bash
python src/green_mlops.py
```

### 3. Output
The script will output latency and emission metrics to the console:
```text
[3/4] ⚡ Running Benchmark: DISTILBERT | Mode: BIO-INSPIRED
   ⏱️  Duration: 0.29s
   🌍 Emissions: 6.72e-06 kgCO2eq
   📉 Avg Latency: 2.90 ms
   🚪 Admission Rate: 58.0%

🏆 RESULTS SUMMARY:
   Bio-Controller Improvement: 42.00% reduction in latency.
```

## 📚 Citation
If you use this work, please cite:

```bibtex
@inproceedings{hamdi2026greenmlops,
  title={Green MLOps: Closed-Loop, Energy-Aware Inference with NVIDIA Triton, FastAPI, and Bio-Inspired Thresholding},
  author={Hamdi, Mustapha},
  booktitle={IEEE Conference},
  year={2026}
}
```

## 🔗 Related Work
- **StructuredDNA** (Hamdi, 2025): [arXiv:2512.08968](https://arxiv.org/abs/2512.08968)
- **SGEMAS** (Hamdi, 2025): [arXiv:2512.14708](https://arxiv.org/abs/2512.14708)

---
*Developed by Mustapha Hamdi, PhD.*
