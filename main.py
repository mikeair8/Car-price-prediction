import pandas as pd
import numpy as np
from RandomForest import RandomForestModel
from XGBoost import XGBM
from Lasso import LassoModel
from LGBM import LightGBMModel
from GBM import GradientBoostingModel
from NN import NeuralNetworkModel

class ModelSelector:
    def __init__(self, data_file):
        self.models = {
            'GBM': GradientBoostingModel,
            'XGBM': XGBM,
            'LGBM': LightGBMModel,
            'RF': RandomForestModel,
            'Lasso': LassoModel,
            'NN': NeuralNetworkModel
        }
        self.data_file = data_file
        self.model = None
        self.X = None
        self.y = None
        self.load_data()

    def load_data(self, to_remove=None):
        if to_remove is None:
            to_remove = ["Pris","URL","Salgsform","Antall dorer", "Farge", "Rekkevidde (WLTP)", "Girkasse", "Avgiftsklasse", "Postnummer", "Karosseri", "Antall seter", "Hjuldrift"]
        df = pd.read_csv(self.data_file)
        self.X = df.drop(columns=to_remove)
        self.y = np.log(df['Pris'])

    def select_model(self, model_name):
        if model_name in self.models:
            self.model = self.models[model_name](self.X, self.y)
        else:
            raise ValueError(f"Model {model_name} not recognized")

    def train_and_evaluate(self, **kwargs):
        if self.model:
            return self.model.train_and_evaluate(**kwargs)
        else:
            raise RuntimeError("No model selected")

# Usage example
selector = ModelSelector('cars_norway_scaled.csv')
selector.select_model('RF')
_, results = selector.train_and_evaluate(use_saved_model=True)
print(results)
