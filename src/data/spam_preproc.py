import pandas as pd
import numpy as np
import sklearn
import torch
import os
import re
from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup
from pathlib import Path
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords


######################
# Create base spam df
######################
def create_df_spam_example(base_path="../datasets/spam_email_raw/"):
    directories = [("spam_2002/", 1), ("spam_2_2003/", 1), ("easy_ham_2002", 0)]

    spams_exs = []
    for directory in directories:
        entries = os.scandir(base_path + directory[0])
        for i, entry in enumerate(entries):
            resulting_dict = clean_file(entry.path)
            if resulting_dict == -1:
                pass
            else:
                resulting_dict['outcome'] = directory[1]
                spams_exs.append(resulting_dict)

    result_df = pd.DataFrame(spams_exs)
    result_df.to_csv("../datasets/spam.csv")


#####################
# Helper functions
#####################
# with some inspiration from
# https://sdsawtelle.github.io/blog/output/spam-classification-part1-text-processing.html
def get_body(path):
    myfile = open(path, encoding="latin-1")
    lines = myfile.readlines()
    try:
        idx = lines.index("\n")  # only grabs first instance
    except:
        try:
            idx = lines.index(r"\\n")  # only grabs first instance
        except:
            return -1
    return "".join(lines[idx:])


def identify_tags(txt, newline):
    from_content = \
        " ".join(re.findall(r"From:(.*?)" + newline, txt))
    to_content = \
        " ".join(re.findall(r"To:(.*?)" + newline, txt))
    subject_content = \
        " ".join(re.findall(r"Subject:(.*?)" + newline, txt))
    date_content = \
        " ".join(re.findall(r"Date:(.*?)" + newline, txt))
    content_encoding = \
        " ".join(re.findall(r"Content-Transfer-Encoding:(.*?)" + newline, txt))

    return {'From': from_content, 'To': to_content, \
            'Subject': subject_content, 'Date': date_content, \
            'Content-Transfer-Encoding': content_encoding}


def process_meta(path):
    email_raw = open(path, encoding="latin-1").read()
    if len(re.findall(r"From:(.*?)\n", email_raw)) > 0:
        header_contents = identify_tags(email_raw, newline=r"\n")
    elif len(re.findall(r"From:(.*?)\\n", email_raw)) > 0:
        header_contents = identify_tags(email_raw, newline=r"\\n")
    else:
        return -1
    return header_contents


def process_body(path):
    body_txt = get_body(path)
    if body_txt == -1:
        return -1
    soup = BeautifulSoup(body_txt, 'html.parser')

    # count html tags before removing
    nhtml_tags = len(soup.findAll())

    # Pull out only the non-markup tex
    body = soup.get_text()

    # Count and remove all URLs
    regx = re.compile(r"(?:(http://[^\s]*|https://[^\s]*))")
    save_urls = re.findall(regx, body)
    body, nurls = regx.subn(repl="", string=body)

    # Count and remove all email addresses
    regx = re.compile(r"\b[^\s]+@[^\s]+[.][^\s]+\b")
    save_emailsadd = re.findall(regx, body)
    body, nemails = regx.subn(repl="", string=body)

    # Count and remove all digits
    regx = re.compile(r"\b[\d.]+\b")
    body, ndigits = regx.subn(repl="", string=body)

    # Count and remove punctuation: ?, !, $
    regx = re.compile(r"[$]")
    body, ndollar = regx.subn(repl="", string=body)
    regx = re.compile(r"[!]")
    body, nexcal = regx.subn(repl="", string=body)
    regx = re.compile(r"[?]")
    body, nquest = regx.subn(repl="", string=body)

    # Count and remove all other punctuation
    regx = re.compile(r"([^\w\s]+)|([_-]+)")
    body, npunc = regx.subn(repl=" ", string=body)

    # Count new lines, replace with special chars
    regx = re.compile(r"\n")
    body, n_new_line = regx.subn(repl=" §§§§ ", string=body)

    # Make all white space a single space, count
    regx = re.compile(r"\s+")
    body, nwhites = regx.subn(repl=" ", string=body)

    # Count number of words
    n_words = len(re.findall(r'\w+', body))

    if n_words == -1:
        return -1

    # get list of sentences, recover specific ones
    all_sent = re.findall(r"\w.*?§§§§", body)
    if len(all_sent) == 0:
        return -1
    elif len(all_sent) == 1:
        first_sent = all_sent[0].replace("§§§§", '')
        second_sent = 'None'
        penult_sent = 'None'
        last_sent = 'None'
    elif len(all_sent) >= 4:
        first_sent = all_sent[0].replace("§§§§", '')
        second_sent = all_sent[1].replace("§§§§", '')
        penult_sent = all_sent[-2].replace("§§§§", '')
        last_sent = all_sent[-1].replace("§§§§", '')
    else:
        first_sent = all_sent[0].replace("§§§§", '')
        second_sent = 'None'
        penult_sent = 'None'
        last_sent = all_sent[-1].replace("§§§§", '')

    return {'nhtml_tags': nhtml_tags, 'save_urls': save_urls, 'nurls': nurls, \
            'save_emailsadd': save_emailsadd, 'nemails': nemails, 'ndigits': ndigits, \
            'ndollar': ndollar, 'nexcal': nexcal, 'nquest': nquest, 'npunc': npunc, \
            'n_new_line': n_new_line, 'nwhites': nwhites, 'n_words': n_words, 'all_sent': str(all_sent), \
            'first_sent': first_sent, 'second_sent': second_sent, 'penult_sent': penult_sent, 'last_sent': last_sent}


def clean_file(path):
    #### One method to rule them all
    # email header procedure
    header_meta_dict = process_meta(path)
    if header_meta_dict == -1:
        return -1
    # email body procedure
    email_body_dict = process_body(path)
    if email_body_dict == -1:
        return -1
    joint_dict = header_meta_dict.copy()
    joint_dict.update(email_body_dict)
    # return single dict with both header and body
    return joint_dict

def clean_spam():
    df = pd.read_csv("../datasets/spam.csv")

    def clean_text(txt):
        try:
            txt = txt.lower()
        except:
            return txt
        regx = re.compile(r"([^\w\s]+)|([_-]+)")
        txt = regx.sub(repl=" ", string=txt)
        txt_words = txt.split(" ")
        filtered_words = [word for word in txt_words if word not in stopwords.words('english')]
        clean_txt = " ".join(filtered_words)
        return clean_txt

    df.drop(["Unnamed: 0", "Date", "Content-Transfer-Encoding", "all_sent"], axis=1, inplace=True)
    df['From'] = df['From'].apply(lambda x: clean_text(x))
    df['To'] = df['To'].apply(lambda x: clean_text(x))
    df['Subject'] = df['Subject'].apply(lambda x: clean_text(x))
    df['save_urls'] = df['save_urls'].apply(lambda x: " ".join(eval(x)))
    df['save_urls'] = df['save_urls'].apply(lambda x: clean_text(x))
    df['save_emailsadd'] = df['save_emailsadd'].apply(lambda x: " ".join(eval(x)))
    df['save_emailsadd'] = df['save_emailsadd'].apply(lambda x: clean_text(x))
    df['first_sent'] = df['first_sent'].apply(lambda x: clean_text(x))
    df['second_sent'] = df['second_sent'].apply(lambda x: clean_text(x))
    df['penult_sent'] = df['penult_sent'].apply(lambda x: clean_text(x))
    df['last_sent'] = df['last_sent'].apply(lambda x: clean_text(x))

    # Only keep string features
    outcome_ind = list((df.columns=='outcome').nonzero()[0])
    counts_inds = list((df.columns.str.startswith('n')).nonzero()[0])
    all_features_inds = list((df.columns!='outcome').nonzero()[0])
    str_and_out_inds = list(set(range(len(df.columns)))-set(counts_inds))
    df_just_str = df.iloc[:, str_and_out_inds]
    df_just_str.to_csv("../datasets/spam_clean_str.csv")

def get_vector_rep(df):
    glove2word2vec(glove_input_file="../datasets/aids/glove.6B/glove.6B.50d.txt",
                   word2vec_output_file="../datasets/aids/glove.6B/gensim_glove_vectors_50d.txt")

    glove_model = KeyedVectors.load_word2vec_format("../datasets/aids/glove.6B/gensim_glove_vectors_50d.txt", binary=False)

    def word_vector(word2vec_model, word):
        try:
            return word2vec_model[word]
        except KeyError:
            return np.nan

    def document_vector(word2vec_model, doc):
        # remove out-of-vocabulary words
        if isinstance(doc, str):
            doc = [word2vec_model[word] for word in doc.split(" ") if word in word2vec_model.vocab]
            if len(doc)>0:
                return np.mean(doc, axis=0)
            else:
                ex_vec = glove_model['apple']
                return np.array([0]*len(ex_vec))
        else:
            ex_vec = glove_model['apple']
            return np.array([0]*len(ex_vec))

    df_for_classification = df.copy()
    df_for_classification['From_glv_vec'] = df_for_classification['From'].apply(lambda x: document_vector(glove_model, x))
    df_for_classification['To_glv_vec'] = df_for_classification['To'].apply(lambda x: document_vector(glove_model, x))
    df_for_classification['Subject_glv_vec'] = df_for_classification['Subject'].apply(lambda x: document_vector(glove_model, x))
    df_for_classification['Urls_glv_vec'] = df_for_classification['save_urls'].apply(lambda x: document_vector(glove_model, x))
    df_for_classification['Emails_glv_vec'] = df_for_classification['save_emailsadd'].apply(lambda x: document_vector(glove_model, x))
    df_for_classification['FirstSent_glv_vec'] = df_for_classification['first_sent'].apply(lambda x: document_vector(glove_model, x))
    df_for_classification['SecondSent_glv_vec'] = df_for_classification['second_sent'].apply(lambda x: document_vector(glove_model, x))
    df_for_classification['PenultSent_glv_vec'] = df_for_classification['penult_sent'].apply(lambda x: document_vector(glove_model, x))
    df_for_classification['LastSent_glv_vec'] = df_for_classification['last_sent'].apply(lambda x: document_vector(glove_model, x))

    # Only use string features
    vector_columns = [c for c in df_for_classification.columns if 'glv_vec' in c]
    count_columns = [c for c in df_for_classification.columns if c.startswith('n')]
    df_for_classification = df_for_classification[vector_columns+['outcome']]
    df_for_classification.to_csv("../datasets/spam_for_classification.csv")

def get_flat_vector_rep(df_for_clf):
    df_for_classification_flatten = df_for_clf.copy()

    def flatten_column(df, c):
        df[[c+str(i) for i in range(len(df[c].iloc[0]))]] =\
            pd.DataFrame(df[c].tolist(), index = df.index)
        df.drop(c, axis=1, inplace=True)
        return df

    for c in df_for_clf.columns[:-1]:
        df_for_classification_flatten = flatten_column(df_for_classification_flatten, c)

    cols = list(df_for_classification_flatten.columns[1:]) + [df_for_classification_flatten.columns[0]]
    df_for_classification_flatten = df_for_classification_flatten[cols]
    df_for_classification_flatten.head()

    df_for_classification_flatten.to_csv("../datasets/spam_for_classification_flat.csv")