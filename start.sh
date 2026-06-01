#!/bin/bash
pkill ollama 2>/dev/null
sleep 1
OLLAMA_NUM_PARALLEL=8 ollama serve &
sleep 2
python3 -m streamlit run src/app.py