import os
from PIL import Image
import requests
import json

def trouver_premiere_image():
    """
    Trouve la première image dans le dossier courant
    
    Returns:
        str: Chemin vers la première image trouvée ou None si aucune image n'est trouvée
    """
    extensions_images = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    dossier_courant = os.path.dirname(os.path.abspath(__file__))
    
    for fichier in os.listdir(dossier_courant):
        if fichier.lower().endswith(extensions_images):
            return os.path.join(dossier_courant, fichier)
    return None

def analyser_image(chemin_image, question="Que voyez-vous sur cette image ?"):
    """
    Envoie une image au serveur FastAPI pour analyse
    
    Args:
        chemin_image (str): Chemin vers l'image à analyser
        question (str): Question à poser sur l'image
    """
    try:
        if not os.path.exists(chemin_image):
            print(f"Erreur : L'image {chemin_image} n'existe pas")
            return

        with open(chemin_image, 'rb') as image_file:
            files = {'file': (os.path.basename(chemin_image), image_file)}
            params = {'question': question}
            
            # Utiliser le port 8000
            url = "https://uqdl3tfgviiozj-8000.proxy.runpod.net/analyze"
            
            print(f"\nTest avec URL : {url}")
            try:
                response = requests.post(
                    url,
                    files=files,
                    params=params,
                    verify=False,
                    timeout=10
                )
                print(f"Status code : {response.status_code}")
                if response.status_code == 200:
                    resultat = response.json()
                    print(f"Réponse du serveur : {resultat['response']}")
                else:
                    print(f"Erreur : {response.status_code} - {response.text}")
            except Exception as e:
                print(f"Erreur avec {url}: {str(e)}")
                    
    except Exception as e:
        print(f"Erreur lors de l'analyse de l'image : {str(e)}")

if __name__ == "__main__":
    # Trouver la première image
    chemin_image = trouver_premiere_image()
    
    if chemin_image:
        print(f"Image trouvée : {chemin_image}")
        analyser_image(chemin_image)
    else:
        print("Aucune image trouvée dans le dossier courant")