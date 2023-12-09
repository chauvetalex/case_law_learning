import streamlit as st
import asyncio

from time import sleep
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

def main_menu():
    if not 'is_task_done' in st.session_state:
        st.session_state.is_task_done = True

    if st.session_state.is_task_done:
        if st.button('Questions about content', type="primary"):
            st.session_state.is_task_done = False
            ask_questions_about_content()
        elif st.button('Questions about title'):
            st.session_state.is_task_done = False
            st.text('In developpement !')
            sleep(5)
            st.session_state.is_task_done = True
            st.rerun()
        elif st.button('Questions about keywords'):
            st.session_state.is_task_done = False
            st.text('In developpement !')
            sleep(5)
            st.session_state.is_task_done = True
            st.rerun()

# Récupérer des données nettoyées.
@st.cache_data
def load_data():
    records = db_manager.get_records_as_dict()
    record_ids = [rec['id'] for rec in records]
    return record_ids, records

def ask_questions_about_content(n_questions:int=2, thresh:float=0.8):

    def reinit_states():

        record_ids, records = load_data()

        DEFAULT_STATES = {
            'rerun': False,
            'id': '',
            'content': '',
            'is_answer_correct': None
        }
        for key,val in DEFAULT_STATES.items():
            st.session_state[key] = val

        if 'idx_question' not in st.session_state:
            st.session_state.idx_question = 0
        else:
            st.session_state.idx_question += 1

        sample = records[randint(0, len(records))]
        st.session_state.id = sample['id']
        st.session_state.content = 'Tips : ' + sample['content']
        st.session_state.correct_answer = sample['content']

    def set_update_user_answer(*args):
        st.session_state.user_answer = user_answer
        st.session_state.content = user_answer

    # Définir la boucle de questions.
    if 'rerun' not in st.session_state or st.session_state.rerun:
        reinit_states()

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col8:
        st.text(st.session_state.idx_question)
    st.text(st.session_state.id)
    user_answer = st.text_area(
        "Type your answer here",
        st.session_state.content,
        height=300
    )
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        submit_answer_btn = \
            st.button("Submit answer", type="primary", on_click=set_update_user_answer, args=user_answer)
    with col2:
        if st.button('Stop', type='primary'):
            st.session_state.is_task_done = True
            st.rerun()
    if submit_answer_btn and st.session_state.is_answer_correct is None:
        # acceptable_answers = db_manager.get_alternative_answers()
        is_answer_correct, cosine, dotprod = \
            scores.is_similar(st.session_state.correct_answer, st.session_state.user_answer, thresh)
        st.text(f"{is_answer_correct}, {cosine}, {dotprod}")
        st.session_state.is_answer_correct = is_answer_correct
        if st.session_state.is_answer_correct is True:
            next_question_btn = st.button("Next question", type="primary", on_click=reinit_states)
        elif st.session_state.is_answer_correct is False:
            user_input = st.text_area('Do you want to add another acceptable answer ?')
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                add_new_answer_btn = st.button("Add new answer", type="primary", on_click=reinit_states)
            with col2:
                skip_btn = st.button("Skip", on_click=reinit_states)
            if add_new_answer_btn and user_input != '':
                db_manager.add_new_answer(st.session_state.id, user_input)
            elif skip_btn:
                pass

def ask_questions_about_title(n_questions:int=2, thresh:float=0.8):

    def reinit_states():

        record_ids, records = load_data()

        DEFAULT_STATES = {
            'rerun': False,
            'id': '',
            'content': '',
            'is_answer_correct': None
        }
        for key,val in DEFAULT_STATES.items():
            st.session_state[key] = val

        if 'idx_question' not in st.session_state:
            st.session_state.idx_question = 0
        else:
            st.session_state.idx_question += 1

        sample = records[randint(0, len(records))]
        st.session_state.id = sample['content']
        st.session_state.content = 'Tips : ' + sample['id']
        st.session_state.correct_answer = sample['id']

    def set_update_user_answer(*args):
        st.session_state.user_answer = user_answer
        st.session_state.content = user_answer

    # Définir la boucle de questions.
    if 'rerun' not in st.session_state or st.session_state.rerun:
        reinit_states()

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col8:
        st.text(st.session_state.idx_question)
    st.text(st.session_state.id)
    user_answer = st.text_area(
        "Type your answer here",
        st.session_state.content,
        height=300
    )
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        submit_answer_btn = \
            st.button("Submit answer", type="primary", on_click=set_update_user_answer, args=user_answer)
    with col2:
        if st.button('Stop', type='primary'):
            st.session_state.is_task_done = True
            st.rerun()
    if submit_answer_btn and st.session_state.is_answer_correct is None:
        # acceptable_answers = db_manager.get_alternative_answers()
        is_answer_correct, cosine, dotprod = \
            scores.is_similar(st.session_state.correct_answer, st.session_state.user_answer, thresh)
        st.text(f"{is_answer_correct}, {cosine}, {dotprod}")
        st.session_state.is_answer_correct = is_answer_correct
        if st.session_state.is_answer_correct is True:
            next_question_btn = st.button("Next question", type="primary", on_click=reinit_states)
        elif st.session_state.is_answer_correct is False:
            user_input = st.text_area('Do you want to add another acceptable answer ?')
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                add_new_answer_btn = st.button("Add new answer", type="primary", on_click=reinit_states)
            with col2:
                skip_btn = st.button("Skip", on_click=reinit_states)
            if add_new_answer_btn and user_input != '':
                db_manager.add_new_answer(st.session_state.id, user_input)
            elif skip_btn:
                pass


if __name__ == '__main__':
    # main_menu()
    # ask_questions_about_content()
    ask_questions_about_title()
