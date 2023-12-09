# Ce module implémente une simple recherche sémantique avec le framework SentenceTransformers.

# La recherche ne peut pas être effectuée sur un corpus complet trop grand car elle impose des calculs trops lourds.
# Une recherche sémantique sur un grand corpus suppose plutôt une approche de retieval (ANN, FAISS, etc.).
# L'implémentation, tèrs basique, utilise ici des mots clés pour réduire le corpus de recherche.

from sentence_transformers import SentenceTransformer, util

import db_manager

# Définir le modèle d'embeddings.
model = SentenceTransformer('msmarco-roberta-base-v3')

# Importer les données.
records = db_manager.get_records_as_dict()
corpus_embeddings = model.encode([rec['content'] for rec in records])

def search(query:str, keyword:str):

    # Filtrer les recherches avec un keyword pour limiter les candidats.
    filt_records = [rec for rec in records if keyword in rec]
    filt_embeddings = model.encode(filt_records)

    # Appeler la fonction dédiée de sentence_transformers
    qresults = util.semantic_search(model.encode(query), filt_embeddings, top_k=5)

    return qresults


if __name__ == '__main__':
    user_input = input()
    results = search(user_input, keyword='responsabilité')
    print(results)
