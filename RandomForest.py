import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
class RandomForestModel:
    def __init__(self, X, y, find_n=False, n_estimators=1000):
        self.X = X
        self.y = y
        self.find_n = find_n
        self.n_estimators = n_estimators if n_estimators is not None else 1000  # Default value
        self.model = None

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def print_metrics(y_true, y_pred, dataset_name='Validation'):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = RandomForestModel.mean_absolute_percentage_error(y_true, y_pred)
        print(f"{dataset_name} set metrics:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR^2: {r2:.2f}\nMAPE: {mape:.2f}%")

    @staticmethod
    def get_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"RMSE": f"{rmse:.2f}", "MAE": f"{mae:.2f}", "R2": f"{r2:.2f}"}

    def find_best_n_estimators(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.25, random_state=42)
        best_mse_val = float('inf')
        best_n_estimators = 0
        for n_estimators in range(500, 5000, 500):
            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            rf.fit(X_train, y_train)
            y_val_pred = rf.predict(X_val)
            mse_val = mean_squared_error(y_val, y_val_pred)
            if mse_val < best_mse_val:
                best_mse_val = mse_val
                best_n_estimators = n_estimators
        self.n_estimators = best_n_estimators
        print(f"Best number of trees: {best_n_estimators}")

    def train_and_evaluate(self, use_saved_model=False):
        if self.find_n:
            self.find_best_n_estimators()

        X_train_full, X_test, y_train_full, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

        if use_saved_model and os.path.exists('best_model_rf.pkl'):
            print("Loading saved model...")
            self.model = joblib.load('best_model_rf.pkl')
        else:
            print("Training new model...")
            self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
            self.model.fit(X_train_full, y_train_full)
            joblib.dump(self.model, 'best_model_rf.pkl')

        datasets = {'Train': (X_train_full, y_train_full), 'Validation': (X_val, y_val), 'Test': (X_test, y_test)}
        for name, (X_set, y_set) in datasets.items():
            y_pred = self.model.predict(X_set)
            self.print_metrics(np.exp(y_set), np.exp(y_pred), dataset_name=name)

        rf_metrics = {}
        datasets = {'Train': (X_train_full, y_train_full), 'Validation': (X_val, y_val), 'Test': (X_test, y_test)}
        for name, (X_set, y_set) in datasets.items():
            y_pred = self.model.predict(X_set)
            rf_metrics[name] = self.get_metrics(np.exp(y_set), np.exp(y_pred))

        # Output predictions for the test set
        y_test_pred = self.model.predict(X_test)
        test_predictions = pd.DataFrame({'Actual': np.exp(y_test), 'Predicted': np.exp(y_test_pred)})
        # test_predictions.to_csv('predicted_vs_actual_RF.csv', index=False)
        print(test_predictions.head())
        return rf_metrics, test_predictions

# Example usage
# rf_model = RandomForestModel(X, y, find_n=True, n_estimators=500)
# rf_model.train_and_evaluate()
