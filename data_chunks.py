# Ce module regroupe les classes et fonction utiles pour créer des chunks de textes.
# Ces chunks peuvent peuvent être utilisés pour alimenter et ajuster des modèles génératifs (LLM).

import pandas as pd
from nltk.tokenize import word_tokenize as nltk_tokenize
from typing import List, Union


def _run_chunking_engine(text_or_tokens:Union[str, list], window:int, window_offset:int) -> List[str]:

    if isinstance(text_or_tokens, str):
        tokens = nltk_tokenize(text_or_tokens)
    else:
        tokens = text_or_tokens

    chunks = []
    i = 0
    while True:
        chunks.append(tokens[i:i+window])
        i += window_offset
        if window > len(tokens)-i:
            chunks.append(tokens[i:])
            break

    return chunks


# Définir une fonction de chunking sur un paragraphe.
def chunk_at_paragraph_level(text:str):
    pass


# Définir une fonction de chunking glissantes sur x paragraphes.
def chunk_at_n_paragraphs_level():
    pass


if __name__ == '__main__':

    text = """
        39. En second lieu, selon l’article 11 de la Déclaration de 1789 : « La libre communication des pensées et des
        opinions est un des droits les plus précieux de l’homme : tout citoyen peut donc parler, écrire, imprimer
        librement, sauf à répondre de l’abus de cette liberté dans les cas déterminés par la loi ». L’article 34 de la
        Constitution dispose : « La loi fixe les règles concernant ... les droits civiques et les garanties fondamentales
        accordées aux citoyens pour l’exercice des libertés publiques ». Sur ce fondement, il est loisible au législateur
        d’instituer des incriminations réprimant les abus de l’exercice de la liberté d’expression et de communication
        qui portent atteinte à l’ordre public et aux droits des tiers. Cependant, la liberté d’expression et de communication
        est d’autant plus précieuse que son exercice est une condition de la démocratie et l’une des garanties du respect
        des autres droits et libertés. Il s’ensuit que les atteintes portées à l’exercice de cette liberté doivent être
        nécessaires, adaptées et proportionnées à l’objectif poursuivi."""

    chunks = _run_chunking_engine(text, window=2, window_offset=1)
    for chunk in chunks:
        print(chunk)
