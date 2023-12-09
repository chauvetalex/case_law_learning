# Ce module contient les éléments pour une recherche semantique approximée utilisant l'index FAISS.

from sentence_transformers import SentenceTransformer, util
import os
import csv
import pickle
import time
import faiss
import numpy as np

import db_manager


model_name = 'all-MiniLM-L12-v2'
model = SentenceTransformer(model_name)
# print(model.max_seq_length)
embedding_size = 384    # Définir la taille des embeddings (dépend du modèle).

embedding_cache_path = f"embeddings-{model_name.replace('/', '_')}.pkl"

#-------------------------------------------------------------------------------
# Calculer les embeddings ou charger d'éventuels embeddings sérialisés.
if not os.path.exists(embedding_cache_path):

    start_time = time.time()
    raw_data = db_manager.get_records_as_dict()

    # TODO Découper les données brutes en chunks.
    corpus_sentences = set([row['content'] for row in raw_data])

    corpus_sentences = list(corpus_sentences)
    print("Encoding the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)
    end_time = time.time()
    print(f'Computin embeddings took {end_time-start_time:2f}s')

    print("Storing embeddings on disk")
    with open(embedding_cache_path, "wb") as file_output:
        pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, file_output)
else:
    print("Loading pre-computed embeddings from disk")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['sentences']
        corpus_embeddings = cache_data['embeddings']

#-------------------------------------------------------------------------------
# Définir les caractéristiques générales de l'index FAISS.
# Selon la taille des données et l'acuité du résultat souhaité, le choix est différent.
# https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

# En dessous de 1M de vecteurs : 4*sqrt(N) to 16*sqrt(N), avec N le nmbre de vecteurs.
# n_clusters = 1024
n_clusters = 16
# Pour la quantification, IndexFlatIP garantit les meilleurs résultats car il ne compresse pas les vecteurs.
quantizer = faiss.IndexFlatIP(embedding_size)
# La metric utilisée est le 'inner product', c'est à dire le produit scalaire
# Ici appliqué après normalisation des vecteurs, cela revient à une similarité cosinus.
index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)

# Définir le nombre de clusters à explorer au search time.
index.nprobe = 3

print("Start creating FAISS index")
# Les vecteurs d'embeddings sont normalisés à une longueur unitaire (pour que le produit scalaire égale la similarité cosinus.)
corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]

# Ajuster/entrainer l'index pour qu'il génère les clusters.
index.train(corpus_embeddings)

# Ajouter les vecteurs à l'index.
index.add(corpus_embeddings)

# TODO Sauvegarder l'index au format pkl.

def run_search_demo_loop():
    """
    Boucle interactive de démonstration.
    """

    top_k_hits = 5         #Output k hits
    print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))

    while True:
        inp_question = input("Please enter a question: ")

        start_time = time.time()

        # La requête doit être vectorisée et normalisée.
        question_embedding = model.encode(inp_question)
        question_embedding = question_embedding / np.linalg.norm(question_embedding)
        question_embedding = np.expand_dims(question_embedding, axis=0)

        # La fonction 'search' retourne une matrice contenant des mesures de similarité et l'id des corpus
        distances, corpus_ids = index.search(question_embedding, top_k_hits)

        # Créer un dict id:similarité pour la première colonne (les requêtes sont vectorisées. Il est donc possible de faire
        # plusieurs requêtes à la fois pour des questions d'efficience)
        hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
        # Trier le dict en fonction des résultats (ordre décroissant).
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        end_time = time.time()

        # Afficher les résultats.
        print("Input question:", inp_question)
        print("Results (after {:.3f} seconds):".format(end_time-start_time))
        for hit in hits[0:top_k_hits]:
            print("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))

        # Evaluer l'index FAISS en comparant les résultats avec une recherche sémantique exacte (recall).
        start_time = time.time()
        correct_hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k_hits)[0]
        correct_hits_ids = set([hit['corpus_id'] for hit in correct_hits])
        end_time = time.time()
        print(f'Exact semantic search performed in {end_time-start_time:.2f}s')

        ann_corpus_ids = set([hit['corpus_id'] for hit in hits])
        if len(ann_corpus_ids) != len(correct_hits_ids):
            print("Approximate Nearest Neighbor returned a different number of results than expected")

        recall = len(ann_corpus_ids.intersection(correct_hits_ids)) / len(correct_hits_ids)
        print("\nApproximate Nearest Neighbor Recall@{}: {:.2f}".format(top_k_hits, recall * 100))

        if recall < 1:
            print("Missing results:")
            for hit in correct_hits[0:top_k_hits]:
                if hit['corpus_id'] not in ann_corpus_ids:
                    print("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))
        print("\n\n========\n")

if __name__ == '__main__':
    run_search_demo_loop()
