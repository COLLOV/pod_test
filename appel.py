import requests

# Au lieu d'utiliser l'IP interne, utilisez l'URL de l'API Gateway
# Le format est : https://[POD_ID]-[PORT].proxy.runpod.net
url = "https://hnd8t5j0io8c9g-8000.proxy.runpod.net/analyze"

# Ouvrir l'image
with open("/Users/GJV/Projet/sandbox/scrap_actu/logo.png", "rb") as f:
    files = {"file": f}
    # Ajouter une question spécifique (optionnel)
    params = {"question": "Que voyez-vous sur cette image ?"}
    
    # Envoyer la requête
    response = requests.post(url, files=files, params=params)

print(response.json()["response"])