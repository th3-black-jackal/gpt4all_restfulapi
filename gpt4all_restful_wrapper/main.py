#!/usr/bin/env python3
import sys
import os
import json
import tempfile
from pathlib import Path
import re
from datetime import datetime
from typing import Annotated
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from llm_utils import *
from config_manager import ConfigManager
try:
    from gpt4all import GPT4All
    GPT4ALL_AVAILBLE = True
except:
    GPT4ALL_AVAILBLE = False
    print("Warning: 'gpt4all' library not found. Install it with pip install gpt4all")

model: Optional[GPT4All] = None
class QueryRequest(BaseModel):
    prompt: Annotated[str, Field(
        description="The text prompt to send to the GPT4All model",
        examples=["Explain the concept of quantum entangelemtn in simple terms"]
    )]
    max_tokens: int = Field(256, description="Maximum number of tokens to generate")
    temp: float = Field(0.7, description="Sampling temperature for creativity")

app = FastAPI(
    titel="GPT4All Wrapper",
    description="A small FastAPI wrapper to communicate with GPT4All SDK",
    version="1.0.0"
    )

@app.on_event("startup")
async def startup_event():
    global model
    if not GPT4ALL_AVAILBLE:
        print("Model initialization skipped due to missing 'gpt4all' library")
        return
    print('--- Starting Model Initialization ---')
    try:
        config_manager = ConfigManager()
        model_config = config_manager.get_model_config()
        model_path = model_config['model_path'] if model_config['model_path'] else None
        if not os.path.exists(model_path):
            print("Warning: Model file not found at: ", model_path)
            print("GPT4All will attempt to download the model now")
        model = GPT4All(
            model_name=model_config['name'],
            model_path=model_path,
            allow_download=True,
            device='gpu'
        )
    except Exception as e:
        print(f"Error during model generation: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during model generation: {e}"
        )
            

@app.post("/query/")
async def execute_prompt(query:QueryRequest):
    response = model.generate(query.prompt, max_tokens=2048)
    print("Response: ", response)
    return {
        'prompt': query.prompt,
        'response': response
    }


