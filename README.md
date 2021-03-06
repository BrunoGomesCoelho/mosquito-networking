mosquito-networking
==============================

Analizing extremelly simple NN and other models for mosquito classification

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └──  src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        └── models         <- Scripts to train models and then use trained models to make
            │                 predictions
            ├── predict_model.py
            └── train_model.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Installation

Use `make create-environment` - we suggest a name of "mosquito-networking" for the env.
Use `make requirements` :)

Make sure to run `python3 -m ipykernel install --user --name=mosquito-networking` if running notebooks as wells (see [here](https://janakiev.com/blog/jupyter-virtual-envs/) for more help).

# Google Colab

I am using google colab to run basic tests on a GPU. This means that to run, you will need to upload:

- The src/ folder, more spefically the .py in data/
- The file in data/interim
- ALL the audio files in data/

All of this respecting the folder of this project.

Check out notebook "0.7-BrunoGomesCoelho-Colab experiment.ipynb" for a basic script.
