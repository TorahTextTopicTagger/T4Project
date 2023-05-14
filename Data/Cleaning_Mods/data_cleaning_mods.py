# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import re

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