import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

class LassoModel:
    def __init__(self, X, y, find_best_alpha=False, alpha=None):
        self.X = X
        self.y = y
        self.find_best_alpha = find_best_alpha
        self.alpha = alpha if alpha is not None else 1.0  # Default value
        self.model = None
        self.scaler = StandardScaler()

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def print_metrics(y_true, y_pred, dataset_name='Dataset'):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = LassoModel.mean_absolute_percentage_error(y_true, y_pred)
        print(f"{dataset_name} set metrics:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR^2: {r2:.2f}\nMAPE: {mape:.2f}%")

    @staticmethod
    def get_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"RMSE": f"{rmse:.2f}", "MAE": f"{mae:.2f}", "R2": f"{r2:.2f}"}

    def find_best_alpha_(self, X_train, y_train, X_val, y_val):

        best_mse_val = float('inf')
        best_alpha = 0
        for alpha in np.logspace(-6, 2, 100):
            lasso = Lasso(alpha=alpha, random_state=42)
            lasso.fit(X_train, y_train)
            y_val_pred = lasso.predict(X_val)
            mse_val = mean_squared_error(y_val, y_val_pred)
            if mse_val < best_mse_val:
                best_mse_val = mse_val
                best_alpha = alpha

        self.alpha = best_alpha
        print(f"Best alpha: {best_alpha}")

    def train_and_evaluate(self, use_saved_model=False):
        # Splitting the data into training, validation, and test sets
        X_train_full, X_test, y_train_full, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

        # Standardize the features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        X_train_full = self.scaler.fit_transform(X_train_full)

        if use_saved_model and os.path.exists('best_model_lasso.pkl'):
            print("Loading saved model...")
            self.model = joblib.load('best_model_lasso.pkl')
        else:
            if self.find_best_alpha:
                self.find_best_alpha_(X_train, y_train, X_val, y_val)

            print("Training new model...")
            self.model = Lasso(alpha=self.alpha, random_state=42)
            self.model.fit(X_train_full, y_train_full)
            joblib.dump(self.model, 'best_model_lasso.pkl')  # Save the model

        # Evaluating the model on the train, validation, and test datasets
        datasets = {'Train': (X_train_full, y_train_full), 'Validation': (X_val, y_val), 'Test': (X_test, y_test)}
        for name, (X_set, y_set) in datasets.items():
            y_pred = self.model.predict(X_set)
            LassoModel.print_metrics(np.exp(y_set), np.exp(y_pred), dataset_name=name)

        lasso_metrics = {}
        datasets = {'Train': (X_train_full, y_train_full), 'Validation': (X_val, y_val), 'Test': (X_test, y_test)}
        for name, (X_set, y_set) in datasets.items():
            y_pred = self.model.predict(X_set)
            lasso_metrics[name] = self.get_metrics(np.exp(y_set), np.exp(y_pred))

        # Output predictions for the test set
        y_test_pred = self.model.predict(X_test)
        test_predictions = pd.DataFrame({'Actual': np.exp(y_test), 'Predicted': np.exp(y_test_pred)})
        print(test_predictions.head())
        return lasso_metrics, test_predictions




# Example usage
# lasso_model = LassoModel(X, y, find_best_alpha=True)
# lasso_model.train_and_evaluate()
