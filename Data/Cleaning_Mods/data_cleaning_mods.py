# -*- coding: utf-8 -*-
"""functions_for_data_cleaning_capstone.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18vI8yV2aT4NfgIWV1rWcaonPou22c2Av
"""

import os
import subprocess
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import re
    

def download_data():
    
    # Uninstall click package
    subprocess.run(['pip', 'uninstall', 'click', '-y'], check=True)

    # Install dvc and dvc-gdrive packages
    subprocess.run(['pip', 'install', 'dvc', 'dvc-gdrive', '-q'], check=True)

    # Set git user email and name globally
    subprocess.run(['git', 'config', '--global', 'user.email', 'ndcantor@mail.yu.edu'], check=True)
    subprocess.run(['git', 'config', '--global', 'user.name', 'ndcantor'], check=True)

    # Clone the GitHub repository
    subprocess.run(['git', 'clone', 'https://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@github.com/ndcantor/torah-topic-text-tagger.git'], check=True)

    # Change the working directory to torah-topic-text-tagger
    os.chdir('torah-topic-text-tagger')

    # Print the contents of the DVC local configuration file
    with open('.dvc/config.local', 'r') as f:
        print(f.read())

    # Pull the latest changes from DVC remote storage
    subprocess.run(['dvc', 'pull'], check=True)

def load_good_data_as_df(file_path):
    with open(file_path) as f:
        data = json.load(f) 
        df = pd.DataFrame(data)
        return df

# takes 'data' parameter from previous lines
def drop_labels(df):
  df= df.drop(labels= ['_id','segment_level','index', 'issue', 'prev_ref','raw_ref', 'prev_author', 'prev_raw_ref','is_sham','ref','match_text_asp','match_text_sef','index_guess','cnt'],axis=1)
  df.dropna()
  return df

def get_topics_with_count_over_n(df,n):
  counts = df['topic'].value_counts()
  df = df[df['topic'].map(counts)>n]
  return df

def return_df_with_drop_labels(good_json_file_path):
    df = load_good_data_as_df(good_json_file_path)
    df = drop_labels(df)
    return df

def return_df_with_n_top_topics(good_json_file_path,n):
    df = load_good_data_as_df(good_json_file_path)
    df = drop_labels(df)
    df = get_topics_with_count_over_n(df,n)
    return df


def remove_english_from_lines(df, text_column_name='text'):
    
  def remove_english(string):
      return re.sub(r'[a-zA-Z]', '', string)

  english_mask = df[text_column_name].str.contains('[a-zA-Z]')
  df.loc[english_mask, text_column_name] = df.loc[english_mask, text_column_name].apply(remove_english)
  return df

def remove_punctuation_from_lines(df, text_column_name='text'):
    # Create a copy of the input dataframe to avoid modifying the original
    new_df = df.copy()
    
    # Define a regular expression pattern to match all punctuation marks
    pattern = r'[^\w\s]'
    
    # Apply the pattern to the specified column
    new_df[text_column_name] = new_df[text_column_name].apply(lambda x: re.sub(pattern, '', x))
    
    return new_df

download_data()












# !pip uninstall click -y
# !pip install dvc dvc-gdrive -q
# !git config --global user.email "ndcantor@mail.yu.edu"
# !git config --global user.name "ndcantor"
# !git clone https://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@github.com/ndcantor/torah-topic-text-tagger.gi
# !cat .dvc/config.local
# !dvc pull -q

from matplotlib import pyplot as plt
import json
import pandas as pd
import numpy as np
import re

def remove_english_from_lines(df, text_column_name='text'):

  def remove_english(string):
      return re.sub(r'[a-zA-Z]', '', string)

  english_mask = df[text_column_name].str.contains('[a-zA-Z]')
  df.loc[english_mask, text_column_name] = df.loc[english_mask, text_column_name].apply(remove_english)
  return df

def remove_punctuation_from_lines(df, text_column_name='text'):
    new_df = df.copy()
    
    pattern = r'[^\w\s]'
    
    new_df[text_column_name] = new_df[text_column_name].apply(lambda x: re.sub(pattern, '', x))
    
    return new_df

import json
import pandas as pd

def load_good_data_as_df(file_path):
  with open(file_path) as f:
    data = json.load(f) 
    df = pd.DataFrame(data)
    return df
def drop_labels(df):
  df= df.drop(labels= ['_id','segment_level','index', 'issue', 'prev_ref','raw_ref', 'prev_author', 'prev_raw_ref','is_sham','ref','match_text_asp','match_text_sef','index_guess','cnt'],axis=1)
  df.dropna()
  return df

def get_topics_with_count_over_n(df,n):
  counts = df['topic'].value_counts()
  df = df[df['topic'].map(counts)>n]
  return df

using = load_good_data_as_df('/content/torah-topic-text-tagger/data/aspaklaria_good.json')
using = using.loc[:, ['text','pm_ref','topic']].dropna()
top_ten_topics = [topic for (topic,idx) in using.topic.value_counts()[:10].items()]

from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def create_single_topic_df(fixed_data, topic, random_state=613):
  using = fixed_data.copy()
  using['is_target'] = np.where(using['topic']==(topic), 1,0)
  positive = using[using['is_target']==1]
  using = using[using['text'].isin(positive['text']) == False]
  negative = using.sample(len(positive.index), random_state=random_state)
  combined = pd.concat([positive, negative], axis=0)
  return combined

def train_and_pred(using,topic):
    single_topic_df = create_single_topic_df(using,topic)

    texts = single_topic_df['text']
    targets = single_topic_df['is_target'].to_numpy()
    train_text, test_text, train_label, test_label = train_test_split(texts.to_numpy(), targets, random_state=613, test_size=0.2, stratify=targets)

    vectorizer = CountVectorizer(max_features=10000, stop_words=None)
    train_bag = vectorizer.fit_transform(train_text)
    test_bag = vectorizer.transform(test_text)

    clf.fit(train_bag, train_label)
    predictions = clf.predict(test_bag)
    return test_label,predictions

def get_results():
  clfs = [MultinomialNB(), LogisticRegression(), KNeighborsClassifier(), RandomForestClassifier(), DecisionTreeClassifier(), XGBClassifier()]
  clf_names = ['MultinomialNB', 'LogisticRegression', 'KNeighborsClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier', 'XGBClassifier']
  topics = {}
  for topic in tqdm(top_ten_topics):
      model_results = {}
      for clf_idx, clf in enumerate(clfs):
          clf_name = clf_names[clf_idx]
          test_label, predictions = train_and_pred(using, topic)
          report = classification_report(test_label, predictions, output_dict=True)
          model_results[clf_name] = {'accuracy': np.round(report['accuracy'],2),
                                    'precision_0': np.round(report['0']['precision'],2),
                                    'recall_0': np.round(report['0']['recall'],2),
                                    'f1-score_0':np.round(report['0']['f1-score'],2),
                                    'precision_1': np.round(report['1']['precision'],2),
                                    'recall_1': np.round(report['1']['recall'],2),
                                    'f1-score_1': np.round(report['1']['f1-score'],2)}
      topics[topic] = pd.DataFrame(model_results).transpose().reset_index().rename(columns={'index': 'model'})
      topics[topic]['topic'] = topic
      topics[topic] = topics[topic][['topic', 'model', 'accuracy', 'precision_0', 'recall_0', 'f1-score_0', 'precision_1', 'recall_1', 'f1-score_1']]
  results = pd.concat(list(topics.values()), axis=0, ignore_index=True)
  return results

#make sure to first call !pip install google-colab
def download_results(results):
  from google.colab import files
  grouped_data = results.groupby('topic')
  for name, group in grouped_data:
      new_df = pd.DataFrame(group.drop(columns='topic'))
      csv_link = f"{name}-classification-reports.csv"
      new_df.to_csv(csv_link)
      files.download(csv_link)


if __name__ == '__main__':
  df = load_good_data_as_df('/content/torah-topic-text-tagger/data/aspaklaria_good.json')
  df = drop_labels(df)
  df = get_topics_with_count_over_n(df,150)

  count1 = 0
  count2 = 0
  moreThanOneWord = []
  values = []
  value_cts = dict((df['topic'].value_counts()))
  for i in df['topic'].unique():
    temp = i.split(' ')
    if len(temp) > 1:
      value = value_cts[i]
      moreThanOneWord.append(i)
      values.append(value)
      count1 += 1
    else:
      count2 += 1
      values.append(value_cts[i])
  print(count1)
  print(count2)

  moreThanOneWord



  pd.DataFrame(moreThanOneWord).to_csv('moreThanOneWord.csv')