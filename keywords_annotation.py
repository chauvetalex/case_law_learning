# Ce module contient les classes et fonctions permettant une annotation rapide de textes avec des mots clés en utilisant
# la recherche sémantique.
# TODO logger les matches pour debugging.
# TODO la technique fonctionne mais il pourrait être utile d'utiliser un embedding plus large pour le anchor_keyword et
# les chunks.

from sentence_transformers import SentenceTransformer, util
from random import randint, randrange
from concurrent.futures import ProcessPoolExecutor
from time import time
import logging
import numpy as np
import db_manager
import data_chunks

local_logger = logging.getLogger('local_log')
local_logger.setLevel(level=logging.DEBUG)

# Définir le modèle d'embedding.
embedding_model = SentenceTransformer('msmarco-roberta-base-v3')

# Charger les données.
records = db_manager.get_records_as_dict()
texts = [rec['content'] for rec in records]


def search_keywords(anchor_keyword:str, sequence:list, threshold:float=0.7, window:int=4):

    # Prétraiter les séquences (supprimer les stopwords, normaliser la ponctuation).

    # Découper les séquences en chunks.
    sequence_chunks = data_chunks._run_chunking_engine(sequence, window=window, window_offset=1)
    local_logger.debug('Chunks :\n', sequence_chunks)

    # Comparer le vecteur du anchor_keyword avec les  vecteurs des chunks.
    # Si la similarité dépasse le seuil défini par l'uitlisateur, ajouter aux keywords.
    anchor_kw_vector = embedding_model.encode(anchor_keyword)
    seq_chunks_matrix = embedding_model.encode(sequence_chunks)
    similarities = util.cos_sim(anchor_kw_vector, seq_chunks_matrix).flatten().numpy()
    sim_indices = np.argwhere(similarities > threshold).flatten().tolist()
    local_logger.debug('Similarities :\n', similarities)

    keywords = []
    for index in sim_indices:
        keywords.append((sequence_chunks[index], similarities[index]))

    return keywords

if __name__ == '__main__':

    user_input = input()

    start = time()
    for text in texts[0:21]:
        results = search_keywords(user_input, text)
        if len(results):
            print(results)
            print(text)
    end = time()
    print(f'Text preprocessed in {end-start:.2f}s')
