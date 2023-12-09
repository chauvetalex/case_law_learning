# Ce module contient des classes et fonctions utiles pour l'extractraction de keywords.
# Le module utilise des fonctions implémentant les approches proposées par différents frameworks.

from nltk.corpus import stopwords
import matplotlib.pyplot as plt

import db_manager

# Implémentation avec Rake.
def extract_kw_with_rake(texts):
    # https://csurfer.github.io/rake-nltk/_build/html/index.html

    from rake_nltk import Rake, Metric

    # Définir une instance de Rake.
    fr_stopwords = stopwords.words('french')
    rake = Rake(
        language='french',
        stopwords=fr_stopwords,
        include_repeated_phrases=False,
        ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO,
        max_length=5
    )

    # Extraire les mots clés.
    rake.extract_keywords_from_sentences(texts)
    keywords = rake.get_ranked_phrases()

    return keywords

# Implémentation avec yake.
def extract_kw_with_yake(texts):
    # http://yake.inesctec.pt/usage.html

    import yake

    # yake fonctionne avec un texte unique et par une liste de textes.
    if isinstance(texts, list):
        texts = texts = ' '.join(texts)

    # Définir une instance de yake.
    fr_stopwords = stopwords.words('french')

    language = 'fr'
    max_ngram_size = 3
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm' # [leve|jaro|seqm]
    windowSize = 1
    numOfKeywords = 40

    yake_kw_extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=deduplication_thresold,
        dedupFunc=deduplication_algo,
        windowsSize=windowSize,
        top=numOfKeywords,
        stopwords=fr_stopwords
    )

    # Extraire les mots clés.
    keywords = yake_kw_extractor.extract_keywords(texts)
    keywords = [kw for kw, _ in keywords]

    return keywords


# Implémentation avec Textacy.
def extract_kw_with_textacy(texts, algo='textrank'):

    #import textacy.ke
    import textacy
    from textacy import *

    # yake fonctionne avec un texte unique et par une liste de textes.
    if isinstance(texts, list):
        texts = texts = ' '.join(texts)

    # Définir un modèle (Textacy est construit sur Spacy).
    en = textacy.load_spacy_lang("en_core_web_sm")

    # Convertir les texts en documents Spacy.
    doc = textacy.make_spacy_doc(texts, lang=en)

    if algo == 'textrank':
        # Extraires les mots clés (Textacy utilise l'algorithme TextRank).
        keywords = [kps for kps, weights in textacy.extract.keyterms.textrank(doc, normalize="lemma")]

    elif algo == 'sgrank':
        # Extract key words and phrases, using SGRank algorithm, as implemented in Textacy
        keywords = [kps for kps, weights in textacy.extract.keyterms.sgrank(doc, normalize="lemma")]
    else:
        raise ValueError('algo must be either "textrank" or "sgrank"')

    return keywords


# Implémentation avec keybert.
def extract_kw_with_keybert(texts):

    from keybert import KeyBERT

    # Définir le modèle KeyBERT.
    fr_stopwords = stopwords.words('french')
    model = 'all-MiniLM-L6-v2'
    kw_model = KeyBERT(model)

    # Extraire les mots clés.
    keywords = kw_model.extract_keywords(
        texts,
        stop_words=fr_stopwords,
        keyphrase_ngram_range=(1, 2),
        use_maxsum=True,
        nr_candidates=20,
        top_n=5
    )

    return keywords



if __name__ == '__main__':
    texts = [rec['content'] for rec in db_manager.get_records_as_dict()]
    keywords = extract_kw_with_keybert(texts[0:15])
    for kw in keywords:
        print(kw)
    print(len(keywords))
