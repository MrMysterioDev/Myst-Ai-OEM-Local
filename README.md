# Myst_Studio

Local AI studio and virtual-world / chatbot platform for **Myst AI — Mr.AI** (fused with **Kai 9000** Soul Memory, Autonomous Heartbeat, Interactive `kai-ui` Cards, and Free MCP Tools).

## Purpose

- Run a local AI virtual world / agentic studio with **$0 cloud investment**.
- Host a local AI chatbot web UI powered by a from-scratch **Mr.AI** neural engine (`MrAIUnifiedBrain`).
- Use **OEM GGUF models** under `02_Myst_Series` for local RTX 4050 GPU inference.
- Auto-promote high-value memories (`hitCount >= 5`) into permanent Soul system prompts and run 30-minute autonomous health/task heartbeats.
- **No Ollama dependency**.

## Hardware / Runtime

- OS: Windows 10/11 64-bit
- Python: 3.12+
- RAM: 16 GB minimum
- Storage: ~10 GB free on external NTFS drive (`E:\Myst_Studio`)
- GPU: NVIDIA RTX 4050 6 GB
- Current driver check: CUDA UMD 13.5

## Architecture

- **Presentation:** HTML chat UI (`Myst AI Mr.AI`) with interactive `kai-ui` telemetry, memory, and tool cards
- **Application:** `MrAIUnifiedBrain` — pure Python (`Mr.AI Pure-Python Neural Engine`) + optional `koboldcpp` GPU RAG (`Mr.AI Neural RAG + RTX 4050 Core`) + Kai Soul Memory Promoter (`hitCount >= 5`)
- **Data:** JSON corpus (`mr_ai_corpus.json`), SQLite gold memory DB (`golden_memory_bank.db`), Kai Soul Memory DB (`myst_pgvector_local.db`), legal web harvester (`legal_web_harvester.py`)
- **Infra:** `koboldcpp.exe` (Port `5001`), GGUF models under `02_Myst_Series`, `cloudflared.exe` Quick Tunnel

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
- Engine modes:
  - `Mr.AI Pure-Python Neural Engine` (when running standalone without Port `5001`)
  - `Mr.AI Neural RAG + RTX 4050 Core` (when `koboldcpp.exe` is active on Port `5001`)
- Kai Fusion Modules:
  - **Soul Memory Promotion (`hitCount >= 5`):** Automatically promotes frequently recalled facts into `06_Skills\promoted-soul-memory\SKILL.md` and `MrAIUnifiedBrain` system prompts.
  - **Autonomous Heartbeat (Every 30 min):** Audits Port `5001`, Port `5005`, OEM GGUF weights, and SQLite databases.
  - **Zero-Key MCP Bridge:** Built-in standard-library connectors for Jina Markdown Reader, Open-Meteo Weather, and Wikipedia REST API.

## OEM Models

- `E:\Myst_Studio\02_Myst_Series\Myst_Nano_2B\Myst-Nano-2B.gguf`
- `E:\Myst_Studio\02_Myst_Series\Myst_Prime_4B\Myst-Prime-4B.gguf`
- `E:\Myst_Studio\02_Myst_Series\Myst_Realm_8B\Myst-Realm-8B.gguf`
- `E:\Myst_Studio\02_Myst_Series\Myst_Architect_7B\Myst-Architect-7B.gguf`

## Requirements

- `requirements.txt` — no external Python packages required for the core server.
- `SYSTEM_REQUIREMENTS.txt` — runtime requirements.

## Current Status

- Build passed.
- All 4 OEM GGUF models verified under `02_Myst_Series`.
- Server tested and responding on `http://127.0.0.1:5005`.
- Kai Soul Memory (`hitCount >= 5`), 30-minute Heartbeat, and MCP tools integrated.
- Autonomous loop running for build/test/review/fix/download cycle.
