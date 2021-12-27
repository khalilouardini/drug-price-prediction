# Drug Price Prediction

## 0. Project Description

The objective is to predict the price for each drug in the test data set - `drugs_test.csv`

I tried to package this codebase as a python library that can be integrated quickly by collaborators (developers) or regular users. 

## 1. Setup
**System requirements**: `Python3`

to install the package, follow these instructions

-   Create a new envrionment with `conda` or `virtualenv`: `conda create -n name_env python=3.7.12`
-   Activate envrionment: `conda activate name_env`
-   Install the package into your envrionement - `pip install -e 'git+https://github.com/khalilouardini/drug_price_prediction.git#egg=drug_price_prediction'`
-   Install requirements - `pip install -r requirements.txt`
-   Make sure `pytest` is installed and run the tests: `python -m pytest` 
-   If **all the tests pass**, you are all set!

## 2. Instructions 

### Running predictions
First make sure you have a directory with all the data - train and test. By default all the data is in `exploration/data/`

We can select hyperparameters from the command line:
-   `--data_dir`: Path to the data directory
-   `--model`: Model to train [Random Forest or XGBoost]
-   `--n_estimators`: Number of estimators in our ensembling model [number of decision trees]
-   `--do_hyperopt`: Whether to run hyperparameter tuning [random search]
-   `--run_inference`: Whether to run inference on test set

[You can also run `drug_price_prediction --help` in the commande line]

An example would be: `drug_price_prediction --data_dir exploration/data/ --model 'RF' --n_estimators 600 --do_hyperopt False --run_inference True` 

## 3. Discussion

### 3.0 About time management
In this section I will discuss my decisions and workflow for this project. As I briefly mentioned in the description, to accomodate the guidelines and the time limit I set to my self, I decided to focus on the productionalization of the code rather than the models and the performance. Complex data pre-processing pipelines are often the pain point when it comes to putting code in production, so I tried to make sure my code would be reliable enough in a production setting. Therefore, I spent a lot of time on the data engineering and its testing. The idea is that most of the predictive modeling part will come from reliable sources like `scikit-learn`. Therefore, we need to be very careful with the pre-processing and feature engineering (i.e the code that we actually write), to ensure robustness in production and reproducibility.

Globally this is how I organized my time:
-   Feature Engineering: ~ 2h
-   Model selection: ~ 1h30
-   Packaging/Refactoring: ~ 3h

I will briefly detail step by step the process.

### 3.1 Exploratory analysis and feature engineering.

The aim of this part is to familiarize with the dataset and make decisions about the feature engineering approach. This analysis is commented (and justified) step by step in the notebook `exploration/data_analysis_and_feature_engineering.ipynb`. The feature engineering steps are summarized below:

-   Log-transform the price to work with a 'gaussian-like' distribution with a wider spread.
-   Convert the dates to integers, and only keep the year.
-   Encode binary features [e.g "approved status"].
-   Encode ordinal features [e.g "reimbursement rate"] 
-   Encode all categorical features.
-   Encode each text feature [i.e description, active ingredients, pharmaceutical company, dosage form and route of administration] with a  Tfidf vectorizer followed by a PCA for dimensionality reduction

To keep the exploratory part separated on jupyter notebooks, all the analysis is done on a separate branch. The code in that branch will later be refactored and merged with the main branch.

### 3.2 Predictive modelling and evaluation.

For the predictive models we keep it simple and start with a Random Forest regressor. Random Forest is a good candidate for a first baseline beacause of its flexibility:
-   Not too intense computationally
-   Easy to interpret [feature importance analysis] with no linearity assumption
-   No need for data scaling or fancy pre-processing
-   With `scikit-learn`, it's a few lines of codes. We want to keep the code clean and simple for a first iteration

This part is also ran on a different notebook in `prediction_random_forest.ipynb`.

We also experiment with another ensembling method, XGBoost that uses **gradient boosting** rather than bagging to learn with a collection of decision trees.

For **evaluation** we report the ***Root Mean Squared Error (RMSE)***, the ***Mean Absolute Error (MAE)***, the ***Mean Aboslute Percentage Error (MAPE)***  and the ***Pearson (r) correlation*** of the target in logscale.

### 3.3 Refactoring

With these two notebooks, we can conclude the exploratory part of this work. The next step was to refactor the code into simple modules that could be easily tested. Each of these modules shoud have a single clear objective. The refactoring is also done in a separate local branch that will be merged once each module is tested. 

These modules are separated in two scripts:
-   `data.py`: each function in this script is a step of the pre-processing pipeline
-   `models.py`: contains code for fitting models (with or without hyperparameter search) and reporting evaluation metrics.``
-   All the tests are saved in the `tests/` fodler.

### 3.4 Production

After merging the refactoring branch with our main, we create one last branch to merge the data pre-processing and the prediction code into a single pipeline. We use all the modules coded so far in the `pipelines.py` script that summarizes all the necessary steps from pre-processing, to training, evaluation and inference in one place.

For convenience, we use the `click`package and add a `command_line.py` file to run all operations from the linux command line.

## 4. Performance

Finally we reach our best performance on the test set (20% of the train set) of:

-   RMSE = 57.5
-   MAE = 14.3
-   MAPE = 50.2
-   r = 0.87

with a tuned Random Forest regressor model with the following hyperparameters:
-   n_estimators = 600
-   max_depth = 100

 We use this same model for our final predictions. When running predictions (**Section 2**), make sure `--run_inference`is set to **True**. The predictions on the test are saved in the `results/` folder. 
