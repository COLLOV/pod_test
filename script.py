import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import io
import requests
from io import BytesIO
import os

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

def telecharger_image(url, nom_fichier):
    """
    Télécharge une image depuis une URL et la sauvegarde localement
    
    Args:
        url (str): URL de l'image à télécharger
        nom_fichier (str): Nom du fichier pour sauvegarder l'image
    """
    try:
        # Télécharger l'image
        reponse = requests.get(url)
        reponse.raise_for_status()
        
        # Ouvrir l'image avec PIL
        image = Image.open(BytesIO(reponse.content))
        
        # Créer le dossier 'images' s'il n'existe pas
        if not os.path.exists('images'):
            os.makedirs('images')
            
        # Sauvegarder l'image
        chemin_complet = os.path.join('images', nom_fichier)
        image.save(chemin_complet)
        print(f"Image sauvegardée avec succès : {chemin_complet}")
        
    except Exception as e:
        print(f"Erreur lors du téléchargement de l'image : {str(e)}")

def creer_image(largeur, hauteur, couleur, nom_fichier):
    """
    Crée une nouvelle image avec les dimensions et la couleur spécifiées
    
    Args:
        largeur (int): Largeur de l'image en pixels
        hauteur (int): Hauteur de l'image en pixels
        couleur (tuple): Couleur RGB (ex: (255, 0, 0) pour rouge)
        nom_fichier (str): Nom du fichier pour sauvegarder l'image
    """
    try:
        # Créer une nouvelle image
        image = Image.new('RGB', (largeur, hauteur), couleur)
        
        # Créer le dossier 'images' s'il n'existe pas
        if not os.path.exists('images'):
            os.makedirs('images')
            
        # Sauvegarder l'image
        chemin_complet = os.path.join('images', nom_fichier)
        image.save(chemin_complet)
        print(f"Image créée avec succès : {chemin_complet}")
        
    except Exception as e:
        print(f"Erreur lors de la création de l'image : {str(e)}")

# Exemple d'utilisation
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
