# DeCoDE

**A deep learning framework for contrastive dimensional embedding**

DeCoDE is designed to identify robust, transdiagnostic brain–behavior dimensions and neurophysiological biotypes by integrating functional connectivity with behavioral measures.

---

Yong Jiao, Xiaoyu Tong, Gregory A. Fonzo, Ian H. Gotlib, Kilian M. Pohl, Theodore D. Satterthwaite, Jing Jiang, Yu Zhang
[**Deep Learning of Brain–Behavior Dimensions Identifies Transdiagnostic Biotypes in Youth with ADHD and Anxiety Disorders**](https://www.biorxiv.org/content/10.1101/2025.10.13.682243v1.abstract)

<div align=center>
<img src="assets/flowchart.png" width="800">
</div>

## Project Structure

```text
.
├── assets/
│   └── flowchart.png                   # Method overview figure
├── models.py                           # cVAE and GCCA model components
├── run_DeCoDE_pipeline.py              # Executable workflow
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Dataset

* **Adolescent Brain Cognitive Development (ABCD) Study**: [https://abcdstudy.org/](https://abcdstudy.org/)
* **Healthy Brain Network (HBN)**: [https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network)

## Dependencies

This project is implemented in Python 3.14.2. Install dependencies via:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the DeCoDE workflow on random matrices:

```bash
python run_DeCoDE_pipeline.py \
  --epochs 5 \
  --batch-size 256 \
  --hidden-dims 256 \
  --latent-dim 32 \
  --alpha 15 \
  --gamma 5 \
  --device gpu
```
