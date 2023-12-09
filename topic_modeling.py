# Ce module contient une implémentation de topic modeling avec BERTopic et SentenceTransformers.
# Ces frameworks permettent une approche modulaire utilisant notamment l'embeddings des principaux tranformers.
# BERTopix implémente aussi des techniques de dimentionalité reduction et de clustering que scikit learn n'implémente pas.



"""Uniform Manifold Approximation and Projection (UMAP) est une technique de reduction de dimension.
https://umap-learn.readthedocs.io/en/latest/
"""

"""hdbscan (Hierarchical Density-Based Spatial Clustering of Applications with Noise) est une technique de
clusteriserisation non supervisée d'un jeu de données.
https://hdbscan.readthedocs.io/en/latest/index.html
"""

import pathlib

from umap import UMAP                                           # Utilisé pour la réduction de dimension.
from hdbscan import HDBSCAN                                     # Utilisé pour le clustering.
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords as nltk_stopwords

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer

import db_manager

SAVED_MODEL_PATH = 'bert_topic_model_saved'
EMBED_MODEL_ID = 'antoinelouis/biencoder-camembert-base-mmarcoFR'

def model_topics():

    # Construire le topic modeling depuis zéro.
    if not pathlib.Path(SAVED_MODEL_PATH).exists():
        ids = []
        passages = []
        for rec in db_manager.get_records_as_dict():
            ids.append(rec['id'])
            passages.append(rec['content'])

        # 1 - Définir le modèle d'embedding.
        embedding_model = SentenceTransformer(EMBED_MODEL_ID)

        # 2 - Réduire la dimension des embeddings.
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine'
        )

        # 3 - Clusteriser les embeddings réduits.
        hdbscan_model = HDBSCAN(
            min_cluster_size=3,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        # 4 - Tokenizer les topics (tous les documents d'un topic sont concaténés puis on crée une représentation BOW)
        fr_stopwords = nltk_stopwords.words('french')
        vectorizer_model = CountVectorizer(stop_words=fr_stopwords)

        # 5 - Générer une représentation des topics à partir de la représentation BOW précédente.
        ctfidf_model = ClassTfidfTransformer()

        # 6 - (Optional) Fine-tune topic representations with
        # a `bertopic.representation` model
        # representation_model = KeyBERTInspired()

        # All steps together
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            #   representation_model=representation_model
            top_n_words=5,
            n_gram_range=(1,1),
            min_topic_size=2,
            calculate_probabilities=True,
            verbose=True
        )
        topics, probs = topic_model.fit_transform(passages)
        topic_model.save(SAVED_MODEL_PATH)

    # Charger un topic modeling existant.
    else:
        topic_model = BERTopic.load(SAVED_MODEL_PATH)

    return topic_model


if __name__ == '__main__':
    print(topic_model.get_topic_info().head(20))
    print(topic_model.get_topics())
