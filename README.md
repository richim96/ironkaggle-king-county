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

## EDA and Data Preparation
We started with a quality check of the 21,613-sale dataset and confirmed that there were no missing values. During preparation, we parsed `date` into a timestamp so time effects could be used numerically, and we removed `id` and `zipcode` from the modeling table: `id` is not a useful predictor and appears multiple times, while `zipcode` was excluded because location is already represented by `lat` and `long`.

In the EDA, we focused on a few illustrative patterns rather than exhaustive feature pruning. Correlation and scatterplot checks showed that sale price rises most clearly with living-area and quality-related variables such as `sqft_living`, `grade`, and nearby living space, while location (`lat`/`long`) also matters. We also noted that `price` is strongly right-skewed and that the square-footage features are visibly collinear, which is part of the reason we moved from a simple linear baseline toward tree-based regression models - which require less tweaking around scale and skewness.

## Model Training
In the training notebook, we narrowed the modeling dataset to the 5,324 homes priced at `$650K+`, then used an 80/20 train-test split to compare two tree-based regression approaches: a `RandomForestRegressor` and a `LGBMRegressor`. We trained the Random Forest with `n_estimators=200`, `max_depth=12`, `min_samples_leaf=3`, and `random_state=96`, which resulted in a solid non-linear baseline without heavy overfitting.

Among the models developed, LightGBM again performed best on the test set. The Random Forest reached test `R² = 0.7633`, `RMSE = $243,415.93`, and `RMSLE = 18.53%`, while LightGBM improved this to test `R² = 0.8317`, `RMSE = $205,267.92`, and `RMSLE = 16.18%`. Based on those results, we would favor boosted tree regression for final price prediction in the `$650K+` segment because it captures the non-linear effects in size, quality, and location better than the simpler ensemble baseline.
