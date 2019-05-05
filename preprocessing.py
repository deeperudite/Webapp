# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import string
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
from rouge import Rouge
from textblob import TextBlob

def clean(df,tasks,columns):

  for column in columns:
      #Lowercase conversion
      df[column] = df[column].apply(lambda x: x.lower())
      print(column+": Converted to lowercase")

      #Tokenization
      df[column] = df[column].apply(word_tokenize)
      df[column] = df[column].apply(lambda x: " ".join(x))
      print(column+": Tokenized")

      if('- ' in tasks):
        #Split a-b into a and b
        df[column] = df[column].str.replace('-',' ')
        print(column+": - Replaced")

      elif('-_' in tasks):
        #Split a-b into a and b
        df[column] = df[column].str.replace('-','_')
        print(column+": - Replaced")

      if('punct' in tasks):
        #Removing punctuations
        df[column] = df[column].str.replace('[^\w\s]',' ')
        print(column+": Removed punctions ")

      if('num' in tasks):
        #Replacing numbers
        df[column] = df[column].str.replace('[0-9]','#')
        print(column+": Replaced Numbers ")

      if('stop' in tasks):
        #Removing Stop Words
        df[column] = df[column].apply(lambda x: " ".join([w for w in x.split() if w not in stop_words]))
        print(column+": StopWords Removed")

      if('lemma' in tasks):
        #Lemmatization - root words
        df[column] = df[column].apply(lambda x: " ".join([lemmatizer.lemmatize(word,pos='v') for word in x.split()]))
        print(column+": Root words Lemmatized")


  print("Null values:",df.isnull().values.any())
  #print(df.head())
  return df

def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))


def get_basic_features(df,columns):

  for column in columns:
    df[column[0]+'_word_count'] = df[column].apply(lambda x: len(str(x).split(" ")))
    print(column+"Word Count Done")
    df[column[0]+'_char_count'] = df[column].str.len()
    print(column+"Char Count Done")
    df[column[0]+'_avg_word'] = df[column].apply(lambda x: avg_word(x))
    print(column+"Avg Word Length Done")

  return df

def tag_part_of_speech(text):
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    pos_list = pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    adjective_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    verb_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return[noun_count, adjective_count, verb_count]

def get_basic_POS(df,columns):

    for column in columns:
      print("Generating Basic POS Features for "+ column)
      df[column[0]+'_nouns'], df[column[0]+'_adjectives'], df[column[0]+'_verbs'] = zip(*df[column].apply(lambda comment: tag_part_of_speech(comment)))

    return df

def get_advanced_POS(df,columns):

    for column in columns:
      print("Generating Advanced POS Features for "+ column)
      df[column[0]+'_nouns_vs_length'] = df[column[0]+'_nouns'] / df[column[0]+'_char_count']
      df[column[0]+'_adjectives_vs_length'] = df[column[0]+'_adjectives'] / df[column[0]+'_char_count']
      df[column[0]+'_verbs_vs_length'] = df[column[0]+'_verbs'] /df[column[0]+'_char_count']
      df[column[0]+'_nouns_vs_words'] = df[column[0]+'_nouns'] / df[column[0]+'_word_count']
      df[column[0]+'_adjectives_vs_words'] = df[column[0]+'_adjectives'] / df[column[0]+'_word_count']
      df[column[0]+'_verbs_vs_words'] = df[column[0]+'_verbs'] / df[column[0]+'_word_count']

    return df

def get_Jaccard(df,columns):
  df['Jaccard'] = df['q_avg_word']
  for i in df.index:
    a = set(df[columns[0]][i].split())
    b = set(df[columns[1]][i].split())
    c = a.intersection(b)
    df['Jaccard'][i] = float(len(c)) / (len(a) + len(b) - len(c))
  return df

def get_TFIDF(df,columns):
  d = []
  for column in columns:
    l = []
    for i in df.index:
      l.append(df[column][i])
    tfidf = TfidfVectorizer(min_df = 1, max_df = 4.5, ngram_range=(1,2))
    features =  tfidf.fit_transform(l)
    d.append( pd.DataFrame( features.todense(), columns=tfidf.get_feature_names()) )
  return d

def get_Rogue(df,columns):
  rouge = Rouge()
  l1=[]
  l2=[]
  for i in df.index:
    l1.append(df[columns[0]][i])
    l2.append(df[columns[1]][i])
  return pd.DataFrame( rouge.get_scores(l1,l2) )

def fab(row, types):
    if(row['r_'+types]!=0):
        return row['s_'+types]/row['r_'+ types]
    else:
        return 0
def get_new_POS1(data):
  data['s_verbs_vs_r_verbs'] = data.apply(lambda x : fab(x,"verbs"), axis = 1)
  data['s_nouns_vs_r_nouns'] = data.apply(lambda x : fab(x,"nouns"), axis = 1)
  data['s_adjectives_vs_r_adjectives'] = data.apply(lambda x : fab(x,"adjectives"), axis = 1)
  data['s_word_count_vs_r_word_count'] = data['s_word_count']/data['r_word_count']
  data['s_nouns_vs_words_vs_r_nouns_vs_words'] = data.apply(lambda x : fab(x, "nouns_vs_words"), axis = 1)
  data['s_verbs_vs_words_vs_r_verbs_vs_words'] = data.apply(lambda x : fab(x, "verbs_vs_words"), axis = 1)
  data['s_adjectives_vs_words_vs_r_adjectives_vs_words'] = data.apply(lambda x : fab(x, "adjectives_vs_words"), axis = 1)
  return data

def get_new_POS2(data):
  data['rs_word_diff'] = data['r_word_count'] - data['s_word_count']
  data['rs_noun_vs_words_diff'] = data['r_nouns_vs_words'] - data['s_nouns_vs_words']
  data['rs_verb_vs_words_diff'] = data['r_verbs_vs_words'] - data['s_verbs_vs_words']
  data['rs_adjectives_vs_words_diff'] = data['r_adjectives_vs_words'] - data['s_adjectives_vs_words']
  return data

def get_question_tags(df):
  tags = ['how', 'what', 'why', 'who', 'which','when','where','whom']
  for tag in tags:
    df[tag+"_flag"] = df['question'].apply(lambda x: bool(x.split().count(tag)))
  return df
