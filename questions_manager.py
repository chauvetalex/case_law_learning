# TODO regrouper question et réponses dans le même module
# TODO utiliser fire pour la manipulation du terminal et ajouter des couleurs.
# TODO créer une api voireun site pour jouer
import subprocess
import answers_manager
import db_manager
from random import randint

from utils.nlp_utils.text_cosine_sim import BasicSklearnTextCosineSim
from db_manager import session
import scores

def ask_questions_about_content(n_questions:int=2, thresh:float=0.8):

    # Récupérer des données nettoyées.
    records = db_manager.get_records_as_dict()
    record_ids = [rec['id'] for rec in records]

    # Définir la boucle de questions.
    i = 0
    while i <= n_questions:
        subprocess.run('clear')
        sample = records[randint(0, len(records))]
        print(sample['content'], '\n')
        print(sample['id'], '\n')
        user_answer = input()
        acceptable_answers = db_manager.get_alternative_answers()
        is_answer_correct, cosine, dotprod = scores.is_similar(sample['content'], user_answer, thresh)
        print('\n', is_answer_correct, cosine, dotprod)

        if is_answer_correct is False:
            user_input = input('Do you want to add another acceptable answer ? y/n (default n)')
            if user_input in ['y', 'yes']:
                new_answer = input('Type new acceptable answer.')
                db_manager.add_new_answer(sample['id'], new_answer)

        i += 1

    subprocess.run('clear')

def ask_questions_about_title(n_questions:int=2, thresh:float=0.8):

    # Récupérer des données nettoyées.
    records = db_manager.get_records_as_dict()

    # TODO Tester les références d'arrêts avec un simple tfidf ou un word_count entrainés sur les titres prétraités.
    record_ids = [rec['id'] for rec in records]
    answer_checker = BasicSklearnTextCosineSim(record_ids)

    # Définir la boucle de questions.
    i = 0
    while i <= n_questions:
        subprocess.run('clear')
        sample = records[randint(0, len(records))]
        print(sample['id'], '\n')
        print(sample['content'], '\n')
        user_answer = input()
        is_answer_correct, cosine, dotprod = scores.is_similar(sample['id'], user_answer, thresh)
        print('\n', is_answer_correct, cosine, dotprod)
        input()
        i += 1

    subprocess.run('clear')


if __name__ == '__main__':
    ask_questions_about_content()
    #ask_questions_about_title()
