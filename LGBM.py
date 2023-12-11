import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class LightGBMModel:
    def __init__(self, X, y, find_best_leaves=False, num_leaves=40):
        self.X = X
        self.y = y
        self.find_best_leaves = find_best_leaves
        self.num_leaves = num_leaves if num_leaves is not None else 40  # Default value
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
        mape = LightGBMModel.mean_absolute_percentage_error(y_true, y_pred)
        print(f"{dataset_name} set metrics:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR^2: {r2:.2f}\nMAPE: {mape:.2f}%")

    @staticmethod
    def get_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"RMSE": f"{rmse:.2f}", "MAE": f"{mae:.2f}", "R2": f"{r2:.2f}"}

    def find_best_num_leaves(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.25, random_state=42)
        best_mse_val = float('inf')
        best_num_leaves = 0
        for num_leaves in range(20, 500, 20):
            lgbm = lgb.LGBMRegressor(num_leaves=num_leaves, learning_rate=0.1, n_estimators=1000, random_state=42)
            lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
            y_val_pred = lgbm.predict(X_val)
            mse_val = mean_squared_error(y_val, y_val_pred)
            if mse_val < best_mse_val:
                best_mse_val = mse_val
                best_num_leaves = num_leaves
        self.num_leaves = best_num_leaves
        print(f"Best number of leaves: {best_num_leaves}")

    def save_model(self, file_name='best_lgbm_model_7.pkl'):
        # Save the trained model to a file
        joblib.dump(self.model, file_name)
        print(f"Model saved as {file_name}")

    @staticmethod
    def load_model(file_name='best_lgbm_model_7.pkl'):
        # Load and return a model from a file
        return joblib.load(file_name)

    def train_and_evaluate(self, use_saved_model=False):
        if self.find_best_leaves:
            self.find_best_num_leaves()

        X_train_full, X_test, y_train_full, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

        if use_saved_model and os.path.exists('best_lgbm_model_7.pkl'):
            print("Loading saved model...")
            self.model = LightGBMModel.load_model('best_lgbm_model_7.pkl')
        else:
            print("Training new model...")
            self.model = lgb.LGBMRegressor(num_leaves=self.num_leaves, learning_rate=0.1, n_estimators=1000, random_state=42)
            self.model.fit(X_train_full, y_train_full)
            self.save_model()


        datasets = {'Train': (X_train_full, y_train_full), 'Validation': (X_val, y_val), 'Test': (X_test, y_test)}
        for name, (X_set, y_set) in datasets.items():
            y_pred = self.model.predict(X_set)
            LightGBMModel.print_metrics(np.exp(y_set), np.exp(y_pred), dataset_name=name)

        lgbm_metrics = {}
        datasets = {'Train': (X_train_full, y_train_full), 'Validation': (X_val, y_val), 'Test': (X_test, y_test)}
        for name, (X_set, y_set) in datasets.items():
            y_pred = self.model.predict(X_set)
            lgbm_metrics[name] = self.get_metrics(np.exp(y_set), np.exp(y_pred))

        # Output predictions for the test set
        y_test_pred = self.model.predict(X_test)
        test_predictions = pd.DataFrame({'Actual': np.exp(y_test), 'Predicted': np.exp(y_test_pred)})
        #test_predictions.to_csv('predicted_vs_actual_LGBM.csv', index=False)
        print(test_predictions.head())
        return test_predictions

# Example usage
# lgbm_model = LightGBMModel(X, y, find_best_leaves=True)
# lgbm_model.train_and_evaluate()


