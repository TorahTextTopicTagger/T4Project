# About the Project
To see an overview of our project, see our GitHub Pages site [here](https://torahtexttopictagger.github.io/T4Project/).

# Repo Structure
The repository is split into 4 directories:

- [Data](Data) | The raw data we were given by sefaria, the cleaned data we produced, and the cleaning script we used to produce it
- [Demo](Demo) | Python scripts and notebooks to run to utilize the models we built for inference
- [Models](Models) | Where we have our saved models, as well as scripts to recreate these models
- [Results](Results) | Summaries (as classification reports) of our findings

# Getting Started Guide

## Intro
The best way to get a feel for the models is to clone the repo and run the [demos](#demos) below. 

You can also check out the [raw data](#raw-data) we used, the [cleaning code](#cleaned-data--cleaning-mods) we used to produce our train/test data, and the [training scripts](#model-generation-scripts) we used to create our models.

## Data

### Raw Data
#### Labeled Data
- The raw labeled data we were given is a large file called `aspaklaria_good.json`. The file is too big for a GitHub repo, but can be downloaded from Google Drive [here](https://drive.google.com/file/d/1y7V0EAoozwltgtsvI-UcmugBs6fnZxsU/view?usp=share_link). 

- This file contains Torah text fragments (paragraph length) and associated topic labels
    - There are over 1000 topic label values, we generally focused on the top 5 most common (למוד, תורה, תפלה, תשובה, ישראל)
    - Some Torah text fragments appear multiple times in the data with different topic labels, meaning the given text is associated with multiple topics

#### Unlabeled Data

- Additionally, we were given raw unlabeled data that contained a dump of Hebrew text from Sefaria and linked sites, called `all_he_text.txt`.  This file is also too big for GitHub, and can be downloaded from Google Drive [here](https://drive.google.com/file/d/1rey81uAe7maZd7OBr9MOpiLzj4g6v0vO/view?usp=share_link)
- We ultimately did not use this file, but it could theoretically prove helpful in creating unsupervised models or embeddings

### Cleaned Data & Cleaning Mods

- The cleaned labeled data that we used for model training and testing is called `good_df.json` and can be found [here](Data/Cleaned%20Data/good_df.json)
- The data cleaning functions used to produce this file (from the raw labeled data) are in `data_cleaining_mods.py` and are found [here](Data/Cleaning_Mods/data_cleaning_mods.py)

## Demos
There are 3 demos available:
1. An inference demo of our logistic regression models, found [here](Demo/logistic_regression_inference_demo.ipynb)
2. An inference demo of our transformer models, found [here](Demo/transformer_inference_demo.ipynb)
3. A more robust demo website of all models to play around with, found [here](Demo/Demo_website.ipynb)

Instructions for each demo can be found within the notebooks themselves. For the website demo you must be able to run a flask app locally, for details see [here](https://flask.palletsprojects.com/en/2.2.x/quickstart/)

## Models

### Our Saved Models
We saved both LogisticRegression and Transformer binary-classification models for the top 5 most common topics:
- The LogisticRegression models are saved in this repo, [here](Models/Saved_Models/LogisticRegression/). Each topic's model has a directory with the topic's name, and within it 2 components, the vectorizer (`vectorizer.pkl`) and the actual model (`model.pkl`)
- The Transformer models are saved in HuggingFace [here](https://huggingface.co/t4-project), following standard HuggingFace directory structure for BERT models. (Note that the models are named with Enlgish transliterations of the Hebrew topics, because of HuggingFace character restrictions.)

### Model Generation Scripts
We include the training scripts used for training both LogisticRegression and Transformer models.
- The script for generating LogisticRegression models creates models for all 5 top topics, and is found [here](Models/Model%20Generation%20Scripts/Train_and_save_logistic_regression_models_for_all_topics.ipynb)
- The script for generating Transformer models creates 3 models for a single topic, using 3 different pre-trained BERT models. The script can be found [here](Models/Model%20Generation%20Scripts/Train_and_save_transformer_models_for_single_topic.ipynb)

# Room for Improvement


