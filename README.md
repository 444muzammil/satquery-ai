# SatQuery AI

Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis.

Developed for Smart India Hackathon (SIH26167) - ISRO.

## Architecture
SatQuery AI uses an agentic orchestration layer that interprets natural language queries, validates modalities (Optical/SAR/Multispectral), and routes them to the appropriate specialist models (EarthDial for VQA/Captioning/SAR, specialized models for Grounding/Change-Detection/Fusion, and a deterministic CV fallback engine).

## Setup (Backend)
1. `cd backend`
2. `python -m venv venv`
3. `source venv/bin/activate`
4. `pip install -r requirements.txt`
5. `cp .env.example .env`

## RunPod Plug-and-Play
SatQuery AI is designed to run the heavy RS VLM (EarthDial) on RunPod.
To activate it:
1. Update `.env` with:
   ```env
   RS_VLM_PROVIDER=runpod
   RS_VLM_ENDPOINT=https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync
   RS_VLM_API_KEY=your_key
   ```
2. If left unconfigured, the system gracefully falls back to deterministic Computer Vision heuristics, openly stating its limitations in semantic reasoning.

## Running the API
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Setup (Frontend)
1. `cd frontend/frontend`
2. `npm install`
3. `npm run dev`
