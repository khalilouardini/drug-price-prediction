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
              prompt='Path to the data directory',
              help='Path to the data directory')
@click.option('--model',
              default='RF',
              prompt='Model to train (Random Forest or XGBoost)',
              help='Model to train (Random Forest or XGBoost)'
              )
@click.option('--n_estimators',
              default=500,
              prompt='Number of estimarors in our ensembling model',
              help='Number of estimarors in our ensembling model'
              )              
def drug_price_prediction(data_dir, model, n_estimators):
    pipelines.run_drug_price_prediction(data_dir, model, n_estimators)