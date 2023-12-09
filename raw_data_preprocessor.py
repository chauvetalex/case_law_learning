from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')


DB = config.get('PATHS', 'DB')
RAW_DATA = config.get('PATHS', 'RAW_DATA')

import re
import pandas as pd
from db_manager import session, engine

data = 'data/test_gaja.md'

def get_clean_data():

    data = RAW_DATA

    with open(data, 'r') as f:
        raw_data = f.read()

    raw_data = raw_data.split('###')

    std_format_rgx = re.compile(r'(?P<title>\*{2}[^\*]+\*{2})(?P<content>[^\>]+?)(?P<raw_text>\>.*)', re.MULTILINE|re.DOTALL)
    no_raw_text_format_rgx = re.compile(r'(?P<title>\*{2}[^\*]+\*{2})(?P<content>[^\>]+?)', re.MULTILINE|re.DOTALL)

    temp_raw_data = []
    temp_errors = []
    for data in raw_data:
        try:
            temp_raw_data.append(std_format_rgx.search(data).groupdict())
        except AttributeError:
            temp_errors.append(data)

    errors = []
    for data in temp_errors:
        try:
            temp_raw_data.append(no_raw_text_format_rgx.search(data).groupdict())
        except AttributeError:
            errors.append(data)

    df = pd.DataFrame(temp_raw_data)
    df['title'] = df['title'].str.replace('*', '')

    df['raw_text'] = df['raw_text'].str.replace('*', '')
    df['raw_text'] = df['raw_text'].str.replace('>', '')
    df['raw_text'] = df['raw_text'].str.replace('\n', '')
    df['raw_text'] = df['raw_text'].str.replace('\\', '')

    df['content'] = df['content'].str.replace('*', '').str.replace('>', '').str.replace('\n', '').str.replace('\\', '')

    df['level'] = 0

    df.rename(columns={'title': 'id'}, inplace=True)

    df.to_sql('tbl_qa_sources', con=engine, if_exists='replace', index=False)

    return df

def break_explode_md_file():

    data = 'data/test_gaja.md'

    def clean_case():
        pass

    def clean_content(text):
        return text.split('>')[0]

    def clean_keywords(text):
        rgx_keywords = re.compile(r'(`{3})(.*?)(`{3})', re.DOTALL|re.MULTILINE)
        match = rgx_keywords.search(text)
        if match:
            return match.expand(r'\2').split(',')
        else:
            None

    def clean_quotes(text):
        rgx_quotes = re.compile(r'>.*?\n', re.DOTALL|re.MULTILINE)
        return '\n'.join(rgx_quotes.findall(text))

    with open(data, 'r') as f:
        raw_md = f.read()

    rgx_case = re.compile(r'(\#{5})([^\#]+)(\#)', re.DOTALL|re.MULTILINE)
    cases = []
    for case in rgx_case.findall(raw_md):
        cases.append({
            'title': case[1].split('\n')[0],
            'keywords': clean_keywords(case[1]),
            'content': clean_content('\n'.join(case[1].split('\n')[1:])),
            'quotes': clean_quotes(case[1])
        })
    return cases


if __name__ == '__main__':
    cases = break_explode_md_file()

    print(cases[0])
