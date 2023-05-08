# T4Project
Table of Contents
* [Introduction](#introduction)
* [Real-World Problem](#real-world-problem)
* [Sefaria Collaboration](#sefaria-collaboration)
* [Multi-Classification Model](#multi-classification-model)
* [Data Sources](#data-sources)
* [DVC Usage](#dvc-usage)
* [EDA and Cleaning](#eda-and-cleaning)
* [Binary Classification Models](#binary-classification-models)
* [Transformer Models](#transformer-models)
* [Model Productionization](#model-productionization)
* [Conclusion](#conclusion)

**Introduction**

<a name="introduction"></a>
Welcome to our project! In this README, you'll find detailed information about the motivation, data sources, cleaning strategies, and various classification models we used to build our multi-classification model.

**Real-World Problem**

<a name="real-world-problem"></a>
(Describe the real-world problem you're trying to solve with this project.)

**Sefaria Collaboration**

<a name="sefaria-collaboration"></a>
Our project has collaborated with Sefaria, a platform for Jewish texts, to utilize their resources and data for building our classification model.

**Multi-Classification Model**

<a name="multi-classification-model"></a>
We've built binary classifiers for various categories and combined them into a multi-classification model to classify texts based on multiple aspects.

**Data Sources**

<a name="data-sources"></a>

Labeled data (for training classifiers)
Aspaklaria data
Sefaria matching texts
Sefaria source sheets
Unlabeled data (for training word embeddings)
Word embeddings files

**DVC Usage**

<a name="dvc-usage"></a>
We used Data Version Control (DVC) to manage and version our datasets and models throughout the project.

**EDA and Cleaning**

<a name="eda-and-cleaning"></a>

**Dataset basics**

(Provide a brief overview of the dataset.)

Appearance of English in the dataset
Punctuation
Roshei teivot (abbreviations)
Cleaning strategies
Variations of roshei teivot cleaning
Variations of English cleaning

**Binary Classification Models**

<a name="binary-classification-models"></a>
We experimented with various binary classification models, including:

Logistic Regression
Word2Vec
Naive Bayes
Random Forest
Decision Trees
Expectation Maximization (EM)

**Transformer Models**

<a name="transformer-models"></a>
We explored the use of Hebrew transformer models due to the limitations of English-based models. The following models were investigated:

AlephBERT
HeBERT
BEREL

**Model Productionization**

<a name="model-productionization"></a>

Demo website
(Provide a link to the demo website showcasing the multi-classification model.)

Multiple heads on a single transformer model
(Explain the implementation of multiple heads on a single transformer model.)

**Conclusion**

<a name="conclusion"></a>
(Summarize the project's main findings, contributions, and future directions.)
