import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import os

class NeuralNetworkModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = None
        self.scaler = StandardScaler()

    @staticmethod
    def print_metrics(y_true, y_pred, dataset_name='Validation'):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{dataset_name} set metrics:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR^2: {r2:.2f}")

    @staticmethod
    def get_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"RMSE": f"{rmse:.2f}", "MAE": f"{mae:.2f}", "R2": f"{r2:.2f}"}

    def build_model(self, input_shape):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        return model

    def train_and_evaluate(self, epochs=100, batch_size=32, use_saved_model=False):
        X_scaled = self.scaler.fit_transform(self.X)
        X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, self.y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

        if use_saved_model and os.path.exists('best_model_nn.h5'):
            print("Loading saved model...")
            self.model = load_model('best_model_nn.h5')
        else:
            print("Training new model...")
            self.model = self.build_model(X_train.shape[1])
            checkpoint = ModelCheckpoint('best_model_nn.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
            self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early_stopping])

        for name, (X_set, y_set) in {'Train': (X_train, y_train), 'Validation': (X_val, y_val), 'Test': (X_test, y_test)}.items():
            y_pred = self.model.predict(X_set)
            self.print_metrics(y_set, y_pred.flatten(), dataset_name=name)

        nn_metrics = {}
        datasets = {'Train': (X_train_full, y_train_full), 'Validation': (X_val, y_val), 'Test': (X_test, y_test)}
        for name, (X_set, y_set) in datasets.items():
            y_pred = self.model.predict(X_set)
            nn_metrics[name] = self.get_metrics(np.exp(y_set), np.exp(y_pred))

        y_test_pred = self.model.predict(X_test)
        test_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred.flatten()})
        #test_predictions.to_csv('predicted_vs_actual_NN.csv', index=False)
        print(test_predictions.head())
        return test_predictions

