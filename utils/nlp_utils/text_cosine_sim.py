
# TODO pour plus de flexibilité, prévoir de passer des vectorizers autres que sklearn
# TODO externaliser le pretraitement avant tokenization.
# TODO  prévoir la possibilité d'ajouter un vectorisation préexistante.
# TODO Passer une fonction à l'initialisation de la fonction pour plus de souplesse.
# TODO prévoir une persistance des données.

import pathlib
import pickle
from typing import Tuple

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

from nltk.corpus import stopwords
from nltk import word_tokenize

import unicodedata

french_stopwords = stopwords.words('french')

def nltk_french_tokenizer(text):
    return word_tokenize(text, language='french')

# Définir une fonction de prétraitement du texte AVANT tokenization
# (abaissement de la casse, substitution de mots, etc.)
def custom_preprocessor(text):
    # Le prétraitement du texte peut aussi intégrer des remplacements de termes ou la suppression d'expressions.
    # L'exlusion des stopwords ne fonctionne quavec des mots individuels.
    text = text.strip().lower()
    nfkd_form = unicodedata.normalize('NFKD', text)
    text = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return text


# [!] Pour fonctionner, les stopwords doivent passer par le même prétraitement que le texte [!]
stopwords = french_stopwords
stopwords = [custom_preprocessor(_) for _ in stopwords]

class BasicSklearnTextCosineSim:

    def __init__(self,
                 texts,
                 preprocessor='strip_accents+lower',
                 tokenizer='sklearn',
                 vectorizer='tfidf',
                 stopwords='french_sklearn',
                 load_pretrained_vectorizer=None,
                 save_vectorizer:str=None,
                 **kwargs):

        self.texts = texts

        # Gérer les arguments par défaut.
        vectorizer_kwargs = {}

        # Si spécifié, charger un vectorizer préentrainé. Vectoriser les textes.
        if load_pretrained_vectorizer and pathlib.Path(load_pretrained_vectorizer).exists():
            self.vectorizer = pickle.load(load_pretrained_vectorizer)
            self.X = self.vectorizer.transform(texts)

        # Sinon, définir les paramètres du vectorizer. Ajuster le vectorizer. Prétraiter les textes.
        else:
            # Définir le preprocessor.
            if preprocessor == 'strip_accents+lower':
                vectorizer_kwargs['strip_accents'] = 'unicode'
                vectorizer_kwargs['lowercase'] = True
            else:
                vectorizer_kwargs['preprocessor'] = preprocessor

            # Définir le tokenizer.
            if tokenizer == 'sklearn':
                pass
            else:
                vectorizer_kwargs['tokenizer'] = tokenizer

            # Définir les stopwords.
            if stopwords == 'french_sklearn':
                pass
            else:
                vectorizer_kwargs['stop_words'] = stopwords

            # Définir le vectorizer.
            if vectorizer.strip().lower() == 'count':
                self.vectorizer = CountVectorizer(**vectorizer_kwargs)
            elif vectorizer.strip().lower() == 'tfidf':
                self.vectorizer = TfidfVectorizer(**vectorizer_kwargs)
            else:
                raise ValueError

            # Prétraiter et vectoriser les textes. Entrainer/ajuster le vectorizer/
            self.X = self.vectorizer.fit_transform(texts)

        # Sauvegarder le vectorizer.
        if save_vectorizer and load_pretrained_vectorizer is None \
            and pathlib.Path(save_vectorizer).parents[0].exists():
            with open(save_vectorizer,'w') as file:
                pickle.dump(self.vectorizer, file)

    def compute_max_cosine_sim(self, text:str, n_results=1) -> list:
        """
        Calcule les n plus grandes simalités cosinus d'un texte.

        Args:
            text (str): _description_
            n_results (int, optional): _description_. Defaults to 1.

        Returns:
            list: _description_
        """

        # Vectoriser le texte à comparer.
        # [!] Le vectorizer sklearn a besoin qu'on lui fournisse un itérable [!]
        x = self.vectorizer.transform([text])
        #x = self.vectorizer.transform(text)

        # Comparer avec les vecteurs existants et générer un vecteur de
        # similarités cosinus.
        sim_mtx = cosine_similarity(self.X, x)

        # Classer les similarités par ordre décroissant.
        sims = np.sort(sim_mtx, axis=0)[::-1]

        # Trier les n résultats attentus.
        best_sims = sims[0:n_results, 0]

        results = []
        for sim in best_sims:

            # Récupérer l'index correspondant à la similarité calculée.
            sim_idx = np.where(sim_mtx == sim)[0][0]

            # Obtenir le texte vectorisé le plus proche avec l'index et la fonction inverse_transform.
            sim_vectorized = self.vectorizer.inverse_transform(
                self.X[sim_idx])

            # Obtenir le texte original le plus proche avec l'index appliqué aux textes originaux.
            sim_original = self.texts[sim_idx]
            results.append((sim, sim_original, sim_vectorized))

        return results

    def has_cosine_sim(self, text:str, thresh_val=0.8) -> Tuple[bool,float,list]:

        """
        Détermine si un texte a une similarité cosinus supérieure ou égale à un seuil donné.

        Returns:
            _type_: _description_
        """

        # Vectoriser le texte à comparer.
        x = self.vectorizer.transform([text])

        # Comparer avec les vecteurs existants et générer un vecteur de similarités cosinus.
        sim_mtx = cosine_similarity(self.X, x)

        # Obtenir la plus grande similarité (la plus proche de 1)
        max_sim = sim_mtx.max()

        # Obtenir l'index du vecteur le plus proche (le plus proche de 1).
        most_sim_idx = sim_mtx.argmax()

        # Obtenir le texte vectorisé le plus proche avec l'index et la fonction inverse_transform.
        most_sim_vetorized = self.vectorizer.inverse_transform(
            self.X[most_sim_idx])

        # Obtenir le texte original le plus proche avec l'index appliqué aux textes originaux.
        most_sim_original = self.texts[most_sim_idx]

        if max_sim > thresh_val:
            return True, max_sim, most_sim_original, most_sim_vetorized
        else:
            return False, max_sim, most_sim_original, most_sim_vetorized

    def compute_pairwise_cosine_sim(self, str_0:str, str_1:str):
        """
        Calcule la similarité cosinus entre 2 chaines.

        Args:
            str_0 (str): _description_
            str_1 (str): _description_

        Returns:
            _type_: _description_
        """
        return cosine_similarity(
            self.vectorizer.transform([str_0]),
            self.vectorizer.transform([str_1])
        )
