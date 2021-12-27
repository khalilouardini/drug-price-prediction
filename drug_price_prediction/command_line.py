import sys
import logging
import click
from drug_price_prediction import pipelines

logging.basicConfig(
    format='[%(asctime)s|%(module)s.py|%(levelname)s]  %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout
)

@click.command()
@click.option('--data_dir',
              type=click.Path(exists=True),
              default='exploration/data/',
              prompt='Path to the data directory',
              help='Path to the data directory')
@click.option('--model',
              default='RF',
              prompt='Model to train (Random Forest or XGBoost)',
              help='Model to train (Random Forest or XGBoost)'
              )
@click.option('--n_estimators',
              default=500,
              prompt='Number of estimators in our ensembling model',
              help='Number of estimators in our ensembling model'
              )          
@click.option('--do_hyperopt',
              prompt='Whether to run hyperparameter tuning (random search)',
              help='Whether to run hyperparameter tuning (random search)'
              )

def drug_price_prediction(data_dir, model, n_estimators, do_hyperopt):
    pipelines.run_drug_price_prediction(data_dir, model, n_estimators, do_hyperopt)
