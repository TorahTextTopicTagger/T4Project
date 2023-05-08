# About the Project
To see an overview of our project, see our GitHub Pages site [here]().

# Repo Structure
The repository is split into 4 directories:

- [Data](Data) | The constant data, like the raw and cleaned training data we were given by sefaria, as well as the cleaning script we used
- [Demo](Demo) | Python scripts and notebooks to run to utilize the models we built for inference
- [Models](Models) | Where we have our saved models, as well as scripts to recreate these models
- [Results](Results) | Summaries (as classification reports) of our findings

# Usage

## Data
We were given 

## Demos
There are 3 demos available:
1. An inference demo of our logistic regression models, found [here](Demo/logistic_regression_inference_demo.ipynb)
2. An inference demo of our transformer models, found [here](Demo/transformer_inference_demo.ipynb)
3. A more robust demo website of all models to play around with, found [here](Demo/Demo_website.ipynb)

Instructions for each demo can be found within the notebooks themselves. For the website demo you must be able to run a flask app locally, for details see [here](https://flask.palletsprojects.com/en/2.2.x/quickstart/)

