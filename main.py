#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 📘 Google Colab: Microservicio con FastAPI + Modelo Qwen + Ngrok

# ✅ 1. Instalación de dependencias
get_ipython().system('pip install -q fastapi uvicorn nest-asyncio pyngrok transformers torch accelerate')


# In[ ]:


# ✅ 2. Cargar el modelo
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto")


# In[ ]:


# ✅ 3. Crear servidor FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nest_asyncio
import uvicorn
from typing import List, Dict, Union
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="LLM API en Colab")

class PromptRequest(BaseModel):
    prompt: Union[str, List[Dict[str, str]]]
    max_tokens: int = 1024

def build_chat_prompt(prompt: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

def extract_assistant_response(full_response: str) -> str:
    # Busca la parte del texto que viene después de 'assistant'
    if "assistant" in full_response:
        return full_response.split("assistant")[-1].strip()
    return full_response.strip()

@app.post("/generate")
def generate_response(req: PromptRequest):
    if isinstance(req.prompt, list):
        prompt_text = build_chat_prompt(req.prompt)
    else:
        prompt_text = req.prompt

    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    result = extract_assistant_response(decoded)
    return {"answer": result}


# ✅ 4. Exponer el servicio con Ngrok
from pyngrok import ngrok, conf
from dotenv import load_dotenv
import os

# Cargar variables del .env
load_dotenv()
ngrok_token = os.getenv("NGROK_AUTH_TOKEN")

ngrok.set_auth_token(ngrok_token)

# Abrir túnel público
public_url = ngrok.connect(7766)
print("🔗 Tu API está disponible en:", public_url)

# Aplicar asyncio patch para Colab
nest_asyncio.apply()

# Iniciar el servidor (bloqueante)
uvicorn.run(app, host="0.0.0.0", port=7766)

