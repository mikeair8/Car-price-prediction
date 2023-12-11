from RandomForest import RandomForestModel
from XGBoost import XGBM
from Lasso import LassoModel
from LGBM import LightGBMModel
from GBM import GradientBoostingModel
from NN import NeuralNetworkModel
import pandas as pd
import numpy as np


to_remove = ["Pris","URL","Salgsform","Antall dorer", "Farge", "Rekkevidde (WLTP)", "Girkasse", "Avgiftsklasse", "Postnummer", "Karosseri", "Antall seter", "Hjuldrift"]
columns_of_X = []
df = pd.read_csv('cars_norway_scaled.csv')
X = df.drop(columns=to_remove[:10])
y = np.log(df['Pris'])

lgbm_model = LightGBMModel(X, y)
lgbm_model.train_and_evaluate()



for i in range (8,len(to_remove)):
    print(i)

    all_results = []
    df = pd.read_csv('cars_norway_scaled.csv')
    X = df.drop(columns=to_remove[:(i+1)])
    y = np.log(df['Pris'])

    #GBM Model
    print("------------ GBM ------------")
    gbm_model = GradientBoostingModel(X, y)
    gbm_metrics, _ = gbm_model.train_and_evaluate()
    all_results.append(("GBM", gbm_metrics))
    print("")

    # XGBoost Model
    print("------------ XGBM ------------")
    xgbm_model = XGBM(X, y)
    xgbm_metrics, _ = xgbm_model.train_and_evaluate()
    all_results.append(("XGBM", xgbm_metrics))
    print("")

    # LightGBM Model
    print("------------ LGBM ------------")
    lgbm_model = LightGBMModel(X, y)
    lgbm_metrics, _ = lgbm_model.train_and_evaluate()
    all_results.append(("LGBM", lgbm_metrics))
    print("")

    # Random Forest Model
    print("------------ RF ------------")
    rf_model = RandomForestModel(X, y)
    rf_metrics, _ = rf_model.train_and_evaluate()
    all_results.append(("RF", rf_metrics))
    print("")

    # Lasso Regression Model
    print("------------ Lasso ------------")
    lasso_model = LassoModel(X, y, find_best_alpha=True)
    lasso_metrics, _ = lasso_model.train_and_evaluate()
    all_results.append(("Lasso", lasso_metrics))
    print("")

    # Neural Network Model
    print("------------ NN ------------")
    nn_model = NeuralNetworkModel(X, y)
    nn_metrics, _ = nn_model.train_and_evaluate(epochs=5000, batch_size=32, use_saved_model=False)
    all_results.append(("NN", nn_metrics))
    print("")

    # Create a DataFrame from the results
    df_data = []
    for model_name, results in all_results:
        row = [model_name]
        for dataset in ["Train", "Validation", "Test"]:
            metrics = results.get(dataset, {})
            row.extend([metrics.get("RMSE", None), metrics.get("MAE", None), metrics.get("R2", None)])
        df_data.append(row)

    df_columns = ["Model", "Train RMSE", "Train MAE", "Train R2", "Validation RMSE", "Validation MAE", "Validation R2", "Test RMSE", "Test MAE", "Test R2"]
    df = pd.DataFrame(df_data, columns=df_columns)

    df.to_excel(f"model_metrics_{i}.xlsx", index=False)

    columns_of_X.append(X.columns.tolist())

columns_df = pd.DataFrame(columns_of_X).transpose()
columns_df.to_csv("columns_of_X_from8.csv",header=[f"Iter_{i}" for i in range(len(to_remove)-1)], index_label='Index')
# Export to Excel

