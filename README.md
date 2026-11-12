# Myst_Studio

100% Free & Open-Source local AI studio, virtual-world engine, and from-scratch neural chatbot platform for **Myst AI — Mr.AI** (`MrAIUnifiedBrain`), fused with **Kai 9000** (Apache-2.0) Soul Memory Promotion and Autonomous Heartbeat.

## Core Principles ($0 Cost & 100% Open Source)

- **$0 Cloud / API Investment:** Runs 100% on local hardware (`E:\Myst_Studio`) with zero paid APIs, zero telemetry, and zero external CDN trackers.
- **From-Scratch `Mr.AI` Neural Network (`train_mrai_from_scratch.py`):**
  - Structures and trains a 2-layer `SiLU` + `RMSNorm` + `Softmax` neural network from scratch using analytic Cross-Entropy backpropagation in pure Python 3.12 standard library.
  - Automatically downloads and trains on real, 100% open-source datasets:
    1. **Databricks Dolly-15k** (`CC BY-SA 3.0` — commercial & non-commercial open instruction dataset)
    2. **Wikipedia REST API** (`CC BY-SA 4.0` — encyclopedic articles)
    3. **arXiv Open-Access API** (`Open Access` — CS & AI research paper abstracts)
    4. **Project Gutenberg / Gutendex** (`Public Domain` — classic literature)
- **OEM GGUF Models (`02_Myst_Series`):** Local RTX 4050 GPU inference via [KoboldCpp](https://github.com/LostRuins/koboldcpp) (`AGPL-3.0`) on port `5001` with **no Ollama dependency**.

## Build, Train From Scratch & Push to GitHub

```powershell
# 1. Install all FOSS modules, fetch real open-source datasets, and train Mr.AI from scratch
python E:\Myst_Studio\fuse_kai_and_push_github.py

# 2. Re-run from-scratch dataset harvesting & neural backpropagation anytime
python E:\Myst_Studio\11_Myst_AI\train_mrai_from_scratch.py

# 3. Push to GitHub (MrMysterioDev/Myst-Ai-OEM-Local)
python E:\Myst_Studio\fuse_kai_and_push_github.py --push
```
