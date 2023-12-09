# Ce module contient les éléments pour un clustering rapide uniquement basé sur la similarité des textes.
# L'algorithme regroupe les textes en fonction de leur proximité et un paramètre permet de forcer le nombre minimal
# d'individus dans un cluster.

from sentence_transformers import SentenceTransformer, util
import time

import db_manager


# Définir le modèle d'embedding.
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('antoinelouis/biencoder-camembert-base-mmarcoFR')

corpus_sentences = [rec['content'] for rec in db_manager.get_records_as_dict()]

print("Encode the corpus. This might take a while")
corpus_embeddings = model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

print("Start clustering")
start_time = time.time()

# Définir les paramètres du clustering.
clusters = util.community_detection(
    corpus_embeddings,
    min_community_size=3,
    threshold=0.75
)

print("Clustering done after {:.2f} sec".format(time.time() - start_time))

for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
    for sentence_id in cluster[0:3]:
        print("\t", corpus_sentences[sentence_id])
    print("\t", "...")
    for sentence_id in cluster[-3:]:
        print("\t", corpus_sentences[sentence_id])
