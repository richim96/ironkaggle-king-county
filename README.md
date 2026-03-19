# King County House Sales Analysis

Exploratory data analysis and predictive modeling on King County, WA house sale prices (May 2014 – May 2015).
Using a dataset of 21 features across ~21,000 transactions, this project identifies the key drivers of house prices
and builds a machine learning model to predict sale prices — with a focus on properties valued at $650K+.

## Installation
Clone the repository, then run `uv sync` to install all the required dependencies locally.
To add missing dependencies, run:
```bash
uv add pandas # for example
```

## Project structure
- `data/`
  - `raw/`: Raw data files, including `king_country_houses.csv`.
  - `processed/`: Processed data files (to train the models).
- `notebooks/`
  - `eda.ipynb`: Notebook for exploratory data analysis and data preparation.
  - `model_training.ipynb`: Notebook for training the machine learning model.
- `pyproject.toml`: Configuration file for the project dependencies and settings.
