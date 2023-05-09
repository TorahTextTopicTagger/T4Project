# :memo: Contents
- :information_source: [About the Project](#about-the-project)
- :bricks: [Repo Structure](#repo-structure)
- :rocket: [Getting Started Guide](#getting-started-guide)
  * [Intro](#intro)
  * [Data](#data)
    + [Raw Data](#raw-data)
      - [Labeled Data](#labeled-data)
      - [Unlabeled Data](#unlabeled-data)
    + [Cleaned Data & Cleaning Mods](#cleaned-data---cleaning-mods)
  * [Demos](#demos)
  * [Models](#models)
    + [Our Saved Models](#our-saved-models)
    + [Model Generation Scripts](#model-generation-scripts)
  * [Results](#results)
  * [Utils](#utils)
    + [Sefaria Slug-ID Mapping](#sefaria-slug-id-mapping)
- :bulb: [Ideas for Further Development](#room-for-further-development)
  * [Improve Existing Models](#improve-existing-models)
  * [Streamlining a Complete Production Pipeline](#streamlining-a-complete-production-pipeline)
  * [Data Acquisition](#data-acquisition)
  * [Possible New Architectures](#possible-new-architectures)

# About the Project
To see an overview of our project, see our GitHub Pages site [here](https://torahtexttopictagger.github.io/T4Project/).

# Repo Structure
The repository is split into 4 directories:

- [Data](#data) | The raw data we were given by sefaria, the cleaned data we produced, and the cleaning script we used to produce it
- [Demo](#demos) | Python scripts and notebooks to run to utilize the models we built for inference
- [Models](#models) | Where we have our saved models, as well as scripts to recreate these models
- [Results](#results) | Summaries (as classification reports) of our findings
- [Utils](#utils) | Some utility helpers

# Getting Started Guide

## Intro
The best way to get a feel for the models is to clone the repo and run the [demos](#demos) below. 

You can also check out the [raw data](#raw-data) we used, the [cleaning code](#cleaned-data--cleaning-mods) we used to produce our train/test data, and the [training scripts](#model-generation-scripts) we used to create some of our models, and the [metrics](#results) of various models.

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

## Results
Here we report the results of our models, including basic sklearn models, XGBoost, and fine-tuned transformers. The results are organized per-topic, for easy model-to-model comparison. The results are in the form of [sklearn classification reports](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html), and include columns for:
- overall accuracy
- precision/recall for the labels 0 and 1, i.e. not a member of the topic and member of the topic respectively.

## Utils
### Sefaria Slug-ID Mapping
Since the labeled training data was from [Aspaklaria](https://www.aspaklaria.info/), and the target of the project is ultimately for use in the [Sefaria Topics](https://www.sefaria.org/topics) space, we provide a mapping from the Aspaklaria topics used in the code to the associated Sefaria slug-ID's. The `python` function for this is [here](Utils/topic_to_slug/convert_topics_to_sefaria_ids.py), and can be run interactively from a CLI [here](Utils/topic_to_slug/convert_topics_to_sefaria_ids_interactive.py).

# Room for Further Development

## Improve Existing Models
There is likely not much room to improve the accuracy of the existing models through hyperparameter tuning, as our attempts to do this seemed futile.

However, one great improvement would be in the transformer models' storage and inference. Currently, each topic has its own model, composed of the pre-trained model, and a head fine-tuned for the given topic. Therefore, there is serious violation of the [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) principle, since the same pre-trained model is persisted N times for N topics. Additionally, at inference time the input is fed through N binary classification models, where again work is repeated since the first many layers of each model is identical (as they inherit from the same pre-trained model). The improvement to space and inference time would be to persist a unified model, based on the pre-trained model, that has N heads for the N topics being classified. Implementing this was beyond the scope of our project, but would surely be a welcome improvement.

## Streamlining a Complete Production Pipeline
To be able to seamlessly incorporate new labeled data, retrain existing models, and use those models for inference, a complete automated pipeline would be desireable, preferably with a cloud-based architecture.

## Data Acquisition


## Possible New Architectures
Our current approach to multi-label classification is essentially a unified interface to multiple binary classifiers. However, this architecture is suboptimal for 2 reasons:
1. This creates a scattered affect, where models are not unified for the given task
2. It is probable that patters inhere in the data that could help with accuracy of topic A given the classification for topic B, and only a model that is predicting both can account for that

Therefore, a true multi-label classifier is desireable. However, this would require much data-engineering to create useful training data, i.e. texts with N labels each for an N-topic problem space.


