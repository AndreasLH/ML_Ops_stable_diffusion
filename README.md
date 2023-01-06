ML_Ops stable diffusion Butterfly Image generation
==============================

A project for the DTU course Machine Learning Operations. The primary focus is on MLOps and not the models specifically

# Project description: Butterfly Image generation
by:
Andreas H: s194235
Andreas H: s194238
Yucheng F: s194241
Christian A: s194255
Malthe A: s194257

## Overall goal of the project
The goal of the project is to train a stable diffusion image generator to generate photorealistic images of butterflies.
## What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics)

|Data specific framework| Training framework |Utility framework|
|:----:|:----:|:----:|
|Huggingface Diffusers|Pytorch lightning|Hydra|

We are using [Huggingface Diffusers](https://github.com/huggingface/diffusers) as our main framework. To code the model and make it simple we will use Pytorch lightning. We are going to use Hydra to configure the models. Furthermore, we will use W&B to log relevant information pertaining to the training of models.
## How to you intend to include the framework into your project
The Huggingface framework provides some convenience functions to load the data that we are going to use.
Pytorch lightning has some tools to evaluate the quality of the reconstructed images ([Inception score](https://torchmetrics.readthedocs.io/en/stable/image/inception_score.html)), which we are going to use. 
## What data are you going to run on (initially, may change)
We are using a dataset called “[Smithsonian butterflies subset](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset)”. The dataset is a subset of 1000 images of butterflies. Additionally, we might also use a similar dataset with flowers.
## What deep learning models do you expect to use
The deep learning model is an unconditional stable diffusion model, developed by Google. The stable diffusion model generates images by learning to remove noise. It is unconditional because it can generate images directly from noise, without an additional input, such as a text prompt.


# Original source
https://huggingface.co/google/ddpm-cifar10-32

# Data
https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset

# To set up repo
```
git pull
dvc pull
pip install -r requirements.txt
```
or use the dockerfile
```docker build -f env.dockerfile . -t env:latest```





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
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

