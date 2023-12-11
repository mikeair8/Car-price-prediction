#Machine Learning for Car Pricing

## Project Overview
This project offers a versatile machine learning framework for predicting car prices, incorporating a range of models like GBM, XGBM, LGBM, RF, Lasso Regression, and NN. It's tailored for robust performance across various data scenarios.

## File Structure
- `main.py`: Core script for model selection and evaluation.
- `get_metrics.py`: Script for assessing model performance.
- `XGBoost.py`, `LGBM.py`, `GBM.py`, etc.: Individual model files.
- `NN.py`: Neural Network model implementation.
- `DataCleaning.py`: Includes functions for data preprocessing.

## Setup and Installation
1. Clone the repository.
2. Install dependencies: `pandas`, `numpy`, `xgboost`, `lightgbm`, `sklearn`, `tensorflow`.
3. Adapt the code to your data, especially any mention of features in main.py and get_metrics.py. new_data_cleaner.py will be more or less useless for your data

## Usage
1. Import `ModelSelector` from `main.py`.
2. Initialize with the data file: `selector = ModelSelector('cars_norway_scaled.csv')`.
3. Select a model, train, and evaluate:
   ```python
   selector.select_model('RF')
   _, results = selector.train_and_evaluate(use_saved_model=True)
   ```
4. `get_metrics.py` can be used to evaluate and compare models.

## Data Cleaning and Transformation
- `clean_kilometer`: Cleans the 'Kilometer' column.
- `apply_saved_mapping_to_new_data`: Applies saved mappings to new data.
- `remove_columns`: Removes specific columns.
- `scale_new_data`: Scales new data using a saved scaler.
- `clean_and_transform_data`: Main function to clean and transform data.

### Example Usage of Data Cleaning
```python
if __name__ == "__main__":
    columns = ['Merke ID', 'Model ID', 'Farge', 'Hjuldrift', 'Karosseri', 'Salgsform', 'Avgiftsklasse', 'Girkasse', 'Drivstoff']
    mapping_filenames = [f'{mapping}_mapping.csv' for mapping in columns]
    new_df = pd.read_csv('cars_norway_raw.csv')
    cleaned_df = clean_and_transform_data(new_df, mapping_filenames, 'scaler.save')
    cleaned_df.to_csv('cleaned_new_data.csv', index=False)
```

## Contributing
Contributions to improve model selection, feature engineering, or performance are welcome. Follow the existing coding style and document changes.

## License
This project is open-sourced under the [MIT License](LICENSE.md).

## Additional Documentation
A detailed project report is available in PDF format, highlighting key aspects of the project including methodology, data analysis, and results.

