import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import io

# Initialiser FastAPI
app = FastAPI()

# Modèle de données pour la requête
class ChatRequest(BaseModel):
    question: str

# Charger le modèle et le tokenizer
def load_model():
    model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', 
                                    trust_remote_code=True,
                                    attn_implementation='sdpa', 
                                    torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', 
                                            trust_remote_code=True)
    return model, tokenizer

model, tokenizer = load_model()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), question: str = "What is in the image?"):
    try:
        # Lire et convertir l'image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Préparer les messages
        msgs = [{'role': 'user', 'content': [image, question]}]
        
        # Générer la réponse
        response = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer
        )
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
