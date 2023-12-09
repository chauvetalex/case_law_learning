# Ce modules contient les éléments pour un recherche semantique utilisant l'index BM25 puis un re-rank.

import subprocess
import string
import os
import pickle
import time

from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from tqdm.autonotebook import tqdm

import db_manager

if not torch.cuda.is_available():
    print("Warning: No GPU found ! Computing might be very slow !")

# Définir le bi-encoder chargé du retrival (semantic search).
bi_encoder_model = 'antoinelouis/biencoder-distilcamembert-mmarcoFR'
bi_encoder = SentenceTransformer(bi_encoder_model)
bi_encoder.max_seq_length = 128     # Tronquer les textes trop longs (la seq max dépend du modèle)

# Définir le cross-encoder, chargé du re-rank des résultats du retriever.
cross_encoder_model = 'antoinelouis/crossencoder-distilcamembert-mmarcoFR'
cross_encoder = CrossEncoder(cross_encoder_model)

#-------------------------------------------------------------------------------
# Créer le corpus d'embeddings avec le bi-encoder.
# Obtenir les données.

# Calculer les embeddings ou charger d'éventuels embeddings sérialisés.
embedding_cache_path = f"embeddings-{bi_encoder_model.replace('/', '_')}.pkl"
if not os.path.exists(embedding_cache_path):

    ids = []
    passages = []
    for rec in db_manager.get_records_as_dict():
        ids.append(rec['id'])
        passages.append(rec['content'])

    start_time = time.time()
    corpus_embeddings = bi_encoder.encode(
        passages,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    end_time = time.time()
    print(f'Computing embeddings took {end_time-start_time:2f}s')

    print("Storing embeddings on disk")
    with open(embedding_cache_path, "wb") as file_output:
        pickle.dump({'ids':ids, 'sentences': passages, 'embeddings': corpus_embeddings}, file_output)
else:
    print("Loading pre-computed embeddings from disk")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        ids = cache_data['ids']
        passages = cache_data['sentences']
        corpus_embeddings = cache_data['embeddings']

#-------------------------------------------------------------------------------
# Définir un index BM25.
# Définir un tokenizer pour l'index BM25.
def bm25_tokenizer(text):
    fr_stopwords = stopwords.words('french')
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in fr_stopwords:
            tokenized_doc.append(token)

    return tokenized_doc

# Tokenizer le corpus pour l'index BM25.
tokenized_corpus = []
for passage in tqdm(passages):
    tokenized_corpus.append(bm25_tokenizer(passage))

# Définir un index BM25.
bm25 = BM25Okapi(tokenized_corpus)

#-------------------------------------------------------------------------------
# Définir une fonction de recherche pour comparer les approches lexical search/semantic search/retrieve-rerank
def interactive_search_loop(top_k=5):

    while True:

        query = input("Input question:\n").strip()

        #---------------------------------------------------------------------------
        # Afficher les résultats avec une simple recherche sémantique avec l'index BM25.
        bm25_scores = bm25.get_scores(bm25_tokenizer(query))
        top_n = np.argpartition(bm25_scores, -5)[-5:]
        bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

        print(f"Top-{top_k} lexical search (BM25) hits\n")
        for hit in bm25_hits[0:top_k]:
            print('\n')
            print(f"\t{hit['score']:.3f}\t{ids[hit['corpus_id']]}")
            print(passages[hit['corpus_id']].replace("\n", " "))


        #---------------------------------------------------------------------------
        # Utiliser la recherche sémantique (retrive + re-rank).
        # Retrieve avec une recherche sur l'index sur les vecteurs calculés par le biencoder.
        question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
        # Envoyer le tensor sur le cuda s'il y a une carte graphique compatible.
        if torch.cuda.is_available():
            question_embedding = question_embedding.cuda()
        hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
        hits = hits[0]  # Get the hits for the first query

        # Re-rank avec le cross-encoder.
        # Le cross-encoder a besoin d'une input à 2 textes : la requête et les réultats du retrieval par le bi-encoder.
        # En créant une liste de paire requete/resultats du retrival, on peut vectoriser le calcul.
        cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
        cross_scores = cross_encoder.predict(cross_inp)

        # Trier les résultats du cross-encoder et les ajouter au tableaux général.
        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]

        # Afficher les résultats classés par le bi-encoder.
        print("\n-------------------------\n")
        print(f"Top-{top_k} Bi-Encoder Retrieval hits (semantic search\n")
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        for hit in hits[0:top_k]:
            print('\n')
            # print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))
            print(f"\t{hit['score']:.3f}\t{ids[hit['corpus_id']]}")
            print(passages[hit['corpus_id']].replace("\n", " "))

        # Afficher les résultats par le cross-encoder.
        print("\n-------------------------\n")
        print(f"Top-{top_k} Cross-Encoder Re-ranker hits (semantic + rerank)\n")
        hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
        for hit in hits[0:top_k]:
            print('\n')
            # print("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']].replace("\n", " ")))
            print(f"\t{hit['cross-score']:.3f}\t{ids[hit['corpus_id']]}")
            print(passages[hit['corpus_id']].replace("\n", " "))

        if input('\nDo you want to do another query ?') in ['n', 'no']:
            break
        subprocess.run('clear')

    subprocess.run('clear')

if __name__ == '__main__':
    interactive_search_loop()
