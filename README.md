# Myst_Studio

100% Free & Open-Source local AI studio, virtual-world engine, and chatbot platform for **Myst AI — Mr.AI** (`MrAIUnifiedBrain`), fused with **Kai 9000** (Apache-2.0) Soul Memory Promotion and Autonomous Heartbeat.

## Core Principles ($0 Cost & 100% Open Source)

- **$0 Cloud / API Investment:** Runs 100% on local hardware (`E:\Myst_Studio`) with zero paid APIs, zero telemetry, and zero proprietary cloud dependencies.
- **100% Free & Open-Source Stack (FOSS):**
  - **Core Server & UI (`MrAIUnifiedBrain`):** Python 3.12 Standard Library (`PSF License`) + self-contained native CSS UI (zero external CDN scripts).
  - **Database & Vector RAG:** Built-in SQLite3 (`Public Domain`) in `WAL` mode.
  - **Local GPU Inference:** Portable [KoboldCpp](https://github.com/LostRuins/koboldcpp) (`AGPL-3.0`) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (`MIT`) — **no Ollama dependency**.
  - **Agent Memory & Heartbeat Architecture:** Adapted from [SimonSchubert/Kai](https://github.com/SimonSchubert/Kai) (`Apache-2.0`).
  - **Open Training Datasets:** Harvests only open-licensed sources — Wikipedia (`CC BY-SA 4.0`), Project Gutenberg (`Public Domain`), arXiv (`Open Access`), and `databricks-dolly-15k` (`CC BY-SA 3.0`).

## Hardware / Runtime

- OS: Windows 10/11 64-bit
- Python: 3.12+
- RAM: 16 GB minimum
- Storage: ~10 GB free on external NTFS drive (`E:\Myst_Studio`)
- GPU: NVIDIA RTX 4050 6 GB (CUDA UMD 13.5 verified)

## Architecture

- **Presentation:** Self-contained zero-CDN HTML/CSS chat UI (`Myst AI Mr.AI`) with interactive `kai-ui` telemetry and Soul Memory cards
- **Application:** `MrAIUnifiedBrain` — pure Python (`Mr.AI Pure-Python Neural Engine`) + optional `koboldcpp` GPU RAG (`Mr.AI Neural RAG + RTX 4050 Core`) + Kai Soul Memory Promoter (`hitCount >= 5`)
- **Data:** JSON corpus (`mr_ai_corpus.json`), SQLite gold memory DB (`golden_memory_bank.db`), Kai Soul Memory DB (`myst_pgvector_local.db`), FOSS web harvester (`legal_web_harvester.py`)
- **Infra:** `koboldcpp.exe` (Port `5001`), open-weight GGUF models under `02_Myst_Series`, optional `cloudflared` (`Apache-2.0`)

## Build / Setup

```powershell
python E:\Myst_Studio\fuse_kai_and_push_github.py
python E:\Myst_Studio\build_mrai_public.py
powershell -ExecutionPolicy Bypass -File E:\Myst_Studio\download_models.ps1
E:\Myst_Studio\start_myst_ai.bat
E:\Myst_Studio\stop.bat
```

## Mr.AI (`MrAIUnifiedBrain`)

- Server: `E:\Myst_Studio\11_Myst_AI\myst_ai_server.py`
- Runs on `http://127.0.0.1:5005`
- Engine: `Mr.AI Pure-Python Neural Engine` or `Mr.AI Neural RAG + RTX 4050 Core`
- GPU inference via `koboldcpp` on port `5001`

## OEM Models (`02_Myst_Series`)

- `E:\Myst_Studio\02_Myst_Series\Myst_Nano_2B\Myst-Nano-2B.gguf`
- `E:\Myst_Studio\02_Myst_Series\Myst_Prime_4B\Myst-Prime-4B.gguf`
- `E:\Myst_Studio\02_Myst_Series\Myst_Realm_8B\Myst-Realm-8B.gguf`
- `E:\Myst_Studio\02_Myst_Series\Myst_Architect_7B\Myst-Architect-7B.gguf`

## Requirements & Open-Source Governance

- `requirements.txt` — no external Python packages required for the core server.
- `SYSTEM_REQUIREMENTS.txt` — runtime requirements.
- `OPEN_SOURCE_LICENSES.md` — full FOSS audit of every component, model, and dataset.
