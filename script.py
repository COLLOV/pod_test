import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

# Initialiser FastAPI
app = FastAPI()

# Modèle de données pour la requête
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

# Charger le modèle et le tokenizer
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-4",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

model, tokenizer = load_model()

def format_chat_prompt(messages):
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        prompt += f"<|im_start|>{role}<|im_sep|>{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant<|im_sep|>"
    return prompt

@app.post("/generate")
async def generate(request: ChatRequest):
    try:
        # Formater le prompt
        prompt = format_chat_prompt(request.messages)
        
        # Tokenizer et générer
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
