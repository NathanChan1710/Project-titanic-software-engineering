# Titanic_project
Ce projet a pour objectif de mettre en œuvre de bonnes pratiques d’ingénierie logicielle appliquées à un projet de data science.  
Il s’appuie sur le jeu de données Titanic, récupéré sur Kaggle,  afin de prédire la survie des passagers du Titanic à partir de leurs caractéristiques.  
Lien Kaggle de la base de données : https://www.kaggle.com/code/alexisbcook/titanic-tutorial

Les objectifs sont les suivants :
- la structuration des scripts Python
- la qualité logicielle
- les tests unitaires
- la documentation
- l’utilisation de Git/GitHub et d’une pipeline CI/CD

## Project Organization COOKIECUTTER

Pour faire la structure de notre projet nous nous sommes basés sur cet exemple :

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         titanic_project and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── titanic_project   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes titanic_project a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
## Équipe - BUT3 SD VCOD groupe 33

Nathan Chan Sing Man : Division des scripts, Vérification des codes + qualité des codes  
Manohy Ratsimba : Mise en place de la structure, Division des scripts, Tests unitaires  
Camille Franceschin : Division des scripts, PEP8, Pipeline, Documentation  
Assia Boudjraf : Pipeline

