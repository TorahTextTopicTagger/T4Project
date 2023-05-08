# %%
topic_to_english = {
    'ישראל' : 'yisrael',
    'תורה' : 'torah',
    'תשובה' : 'teshuva',
    'תפלה' : 'tefillah',
    'למוד' : 'limmud'
}
def get_transformer_model_path(topic, base_model_name):
  return f't4-project/{topic_to_english[topic]}-{base_model_name}'

def get_logistic_regression_model_path(topic):
  return f'../Models/Saved_Models/LogisticRegression/{topic}/'

BEREL_BASE_PATH = '../Models/Saved_Models/Transformers/BEREL_base'

# %%
from flask import Flask, render_template, request, url_for, flash, redirect, jsonify
import pickle
import numpy as np
import pandas as pd
import os, re

# %%


# %%
from transformers import AutoModelForSequenceClassification, TextClassificationPipeline
from transformers import BertTokenizerFast

# %%
from transformers import BertTokenizer, BertForMaskedLM
from rabtokenizer import RabbinicTokenizer
from transformers import AutoTokenizer

# %%
transformer_topics = ['ישראל', 'למוד', 'תורה', 'תפלה', 'תשובה']

alephBERT_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
berel_tokenizer = RabbinicTokenizer(BertTokenizer.from_pretrained(os.path.join(BEREL_BASE_PATH, 'vocab.txt'), model_max_length=512))
heBERT_tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT", model_max_length=512)

alephBERT_model_pipes = {}
berel_model_pipes = {}
heBERT_model_pipes = {}
for topic in transformer_topics:
    print(get_transformer_model_path(topic, 'alephBERT'))
    alephBERT_loaded_model = AutoModelForSequenceClassification.from_pretrained(get_transformer_model_path(topic, 'alephBERT'), num_labels=2)
    alephBERT_pipe = TextClassificationPipeline(model=alephBERT_loaded_model, tokenizer=alephBERT_tokenizer, return_all_scores=True)
    alephBERT_model_pipes[topic] = alephBERT_pipe

    berel_loaded_model = AutoModelForSequenceClassification.from_pretrained(get_transformer_model_path(topic, 'BEREL'), num_labels=2)
    berel_pipe =TextClassificationPipeline(model=berel_loaded_model, tokenizer=berel_tokenizer, return_all_scores=True)    
    berel_model_pipes[topic] = berel_pipe

    heBERT_loaded_model = AutoModelForSequenceClassification.from_pretrained(get_transformer_model_path(topic, 'heBERT'), num_labels=2)
    heBERT_pipe = TextClassificationPipeline(model=heBERT_loaded_model, tokenizer=heBERT_tokenizer, return_all_scores=True)    
    heBERT_model_pipes[topic] = heBERT_pipe


# %%
def remove_english_from_lines(df, text_column_name='text'):

  def remove_english(string):
      return re.sub(r'[a-zA-Z]', '', string)

  english_mask = df[text_column_name].str.contains('[a-zA-Z]')
  df.loc[english_mask, text_column_name] = df.loc[english_mask, text_column_name].apply(remove_english)
  return df

# %%
def remove_punctuation_from_lines(df, text_column_name='text'):
    # Create a copy of the input dataframe to avoid modifying the original
    new_df = df.copy()
    
    # Define a regular expression pattern to match all punctuation marks
    pattern = r'[^\w\s]'
    
    # Apply the pattern to the specified column
    new_df[text_column_name] = new_df[text_column_name].apply(lambda x: re.sub(pattern, '', x))
    
    return new_df

# %%
cats = [ u'למוד' , u'תורה' , u'תפלה' , u'תשובה' , u'ישראל']


# %%
# load LinReg models
linear_regression_models = {}

for c in cats:
    path = get_logistic_regression_model_path(c)
    model = pickle.load(open(path+'model.pkl', 'rb'))
    vectorizer = pickle.load(open(path+'vectorizer.pkl', 'rb'))
    linear_regression_models[c] = {
        'model' : model,
        'vectorizer' : vectorizer
    }

# %%
def infer_log_reg(topic, text):
    model = linear_regression_models[topic]['model']
    vectorizer = linear_regression_models[topic]['vectorizer']
    text = vectorizer.transform([text])
    pred = model.predict(text)[0]
    prob = model.predict_proba(text)[0]
    return pred, prob


# %%
class Results():
    def __init__(self, topic, model, prob_0, prob_1, userProb ):
        self.topic = topic
        self.model = model
        self.prob_0 = prob_0
        self.prob_1 = prob_1
        self.result = 1 if prob_1>userProb else 0


# %%
app = Flask(__name__,)

@app.route('/', methods=['GET', 'POST'])
def home():
  if request.method=="GET":
      return render_template('input.html')
  
  sentence = request.args.get('sentence',type= str) or request.form.get('sentence', type=str)
  prob = request.args.get('number', type = float) or request.form.get('number', type = float)
  if not prob:
     prob = 50
  prob = prob /100.0
  df = pd.DataFrame()
  df['text']= [sentence]
  df = remove_punctuation_from_lines(df)
  df = remove_english_from_lines(df)
  sentence = ''.join(df['text'])
  if not sentence:
       return render_template('input.html', sentence = ["No Hebrew Text Found"])

  
  scores =[]
  

  for cat in cats:
    lg = infer_log_reg(cat, sentence)
    p0 =lg[1][0]
    p1 =lg[1][1]
    scores.append(Results(cat, "Logistic Regression" ,p0,p1, prob))
    alephBERT = alephBERT_model_pipes[cat](sentence)[0]
    p0 =alephBERT[0]['score']
    p1 =alephBERT[1]['score']
    scores.append(Results(cat, "alephBERT" ,p0,p1, prob))
    berel = berel_model_pipes[cat](sentence)[0]
    p0 =berel[0]['score']
    p1 =berel[1]['score']
    scores.append(Results(cat, "berel" ,p0,p1, prob))
    heBERT = heBERT_model_pipes[cat](sentence)[0]
    p0 =heBERT[0]['score']
    p1 =heBERT[1]['score']
    scores.append(Results(cat, "heBERT" ,p0,p1, prob))
  

  

  return render_template('input.html', scores =scores, sentence = ["Text: "+ sentence + " with probability: " + str(prob)])

# %%
if __name__ == '__main__':
   app.run()


