# Ce module contient des fonctions pour calculer les scores de similarité entre deux chaines de caractères.
# Il sert à calculer la proximité des réponses fournies avec les réponses attendues.

from sentence_transformers import SentenceTransformer, util

# Définir le modèle utilisé pour l'embedding.
# 'all-MiniLM-L6-v2' produit des vecteurs normalisés qui rend le produit scalaire équivalent à la similarité cosinus.
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('antoinelouis/biencoder-camembert-base-mmarcoFR')

def is_similar(sentence0:str, sentence1:str, thresh:float=0.8, mandatory_tokens:list=None) -> bool:
    """
    Calcule un score en utilisant la similarité cosinus et le produit scalaire, et retourne True ou False en fonction d'un
    seuil déterminé par l'utilisateur.
    """

    # Calculer la similarité cosinus.
    cosine_sim = util.cos_sim(
        model.encode(sentence0),
        model.encode(sentence1)
    )

    # Calculer le produit scalaire.
    dotprod = util.dot_score(
        model.encode(sentence0),
        model.encode(sentence1)
    )

    # Vérifier la présence de token obligatoires.
    has_mandatory_tokens = True
    if mandatory_tokens:
        has_mandatory_tokens = set(mandatory_tokens).issubset(sentence1.split())

    return (True, cosine_sim, dotprod) if cosine_sim >= thresh and has_mandatory_tokens else (False, cosine_sim, dotprod)
