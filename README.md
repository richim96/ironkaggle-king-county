# King County House Sales Analysis

Exploratory data analysis and predictive modeling on King County, WA house sale prices (May 2014 – May 2015).
Using a dataset of 21 features across ~21,000 transactions, this project identifies the key drivers of house prices
and combines hedonic and ensemble models to predict sale prices — with a focus on properties valued at $650K+.

## Installation
Clone the repository, then run `uv sync` to install all the required dependencies locally.
To add missing dependencies, run:
```bash
uv add pandas # for example
```

## Project structure
- `data/`
  - `raw/`: Raw data for analysis `king_country_houses.csv`.
  - `processed/`: Processed data files (to train the models).
- `notebooks/`
  - `eda.ipynb`: Notebook for exploratory data analysis and data preparation.
  - `hedonic_price_modeling.ipynb`: Notebook for hedonic price modeling, log transformations, and coefficient analysis.
  - `ensemble_price_modeling.ipynb`: Notebook for tree-based ensemble price prediction focused on `$650K+` homes.
- `pyproject.toml`: Configuration file for the project dependencies and settings.

## EDA and Data Preparation
We started with a quality check of the 21,613-sale dataset and confirmed that there were no missing values. During preparation, we parsed `date` into a timestamp so time effects could be used numerically, and we removed `id` and `zipcode` from the modeling table: `id` is not a useful predictor and appears multiple times, while `zipcode` was excluded because location is already represented by `lat` and `long`.

In the EDA, we focused on the main relationships in the data before moving on to modeling. Correlation and scatterplot checks showed that sale price rises most clearly with living-area and quality-related variables such as `sqft_living`, `grade`, and `sqft_above`, while location (`lat`/`long`) also matters. We also noted that `price` is strongly right-skewed and that the square-footage features are visibly collinear, which motivated the log transformations used later in the hedonic notebook and the move toward tree-based models that are less sensitive to scale and skewness.

## Model Training
In `hedonic_price_modeling.ipynb`, we built a hedonic regression baseline by modeling `log_price` with log-transformed size features and structural, neighborhood, and location variables. That model reached test `RMSE = 0.258` in log-price space (about `$202,613` in dollar terms) with test `R² = 0.767`, and the largest positive coefficients were attached to `lat`, `waterfront`, `log_sqft_living15`, `log_sqft_above`, and `grade`.

In `ensemble_price_modeling.ipynb`, we narrowed the modeling dataset to the 5,324 homes priced at `$650K+`, then used an 80/20 train-test split to compare two tree-based regression approaches: a `RandomForestRegressor` and a `LGBMRegressor`. The Random Forest reached test `R² = 0.7633`, `RMSE = $243,415.93`, and `RMSLE = 18.53%`, while LightGBM improved this to test `R² = 0.8317`, `RMSE = $205,267.92`, and `RMSLE = 16.18%`. Based on those results, we would favor boosted tree regression for final price prediction in the `$650K+` segment because it captures the non-linear effects in size, quality, and location better than the simpler ensemble baseline.
