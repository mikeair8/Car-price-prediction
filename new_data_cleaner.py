import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Data Cleaning Functions
def clean_kilometer(row):
    if pd.isna(row['Kilometer']):
        if row['Salgsform'] == 'Nybil til salgs':
            return 0  # Replace NaN with 0 for new cars
        else:
            return pd.NA  # Keep NaN for used cars to drop later
    else:
        # Remove 'km' and convert to integer if not NaN
        return int(row['Kilometer'].replace(' km', '').replace('.', '').strip())

def apply_saved_mapping_to_new_data(new_df, column, filename):
    # Load the mapping from the CSV file
    mapping_df = pd.read_csv(filename)
    mapping_dict = pd.Series(mapping_df.Mapped_ID.values, index=mapping_df.Original_ID).to_dict()

    # Apply the mapping to the new data
    new_df[column] = new_df[column].map(mapping_dict).fillna(pd.NA)  # Fill unmapped values with NA
    return new_df

def remove_columns(df):
    columns_to_remove = ['Sylindervolum', 'CO2-utslipp', 'Merke', 'Model']
    df = df.drop(columns=columns_to_remove, errors='ignore')
    return df

def scale_new_data(new_data, scaler_filename):
    # Load the scaler from the file
    features_to_scale = ['Effekt', 'Rekkevidde (WLTP)', 'Kilometer']
    scaler = joblib.load(scaler_filename)

    # Scale the new data
    new_data[features_to_scale] = scaler.transform(new_data[features_to_scale])
    return new_data

# Main Cleaning Function
def clean_and_transform_data(df, mapping_filenames, scaler_filename):
    # Clean data
    df['Kilometer'] = df.apply(clean_kilometer, axis=1)
    df.dropna(subset=['Kilometer'], inplace=True)

    # Apply mappings
    for col, mapping_file in zip(columns, mapping_filenames):
        df = apply_saved_mapping_to_new_data(df, col, mapping_file)

    df = remove_columns(df)

    # Scale data
    df = scale_new_data(df, scaler_filename)

    return df

# Example Usage
if __name__ == "__main__":
    columns = ['Merke ID', 'Model ID', 'Farge', 'Hjuldrift', 'Karosseri', 'Salgsform', 'Avgiftsklasse', 'Girkasse', 'Drivstoff']

    mapping_filenames = [f'{mapping}_mapping.csv' for mapping in columns]

    new_df = pd.read_csv('cars_norway_raw.csv')
    cleaned_df = clean_and_transform_data(new_df, mapping_filenames, 'scaler.save')
    cleaned_df.to_csv('cleaned_new_data.csv', index=False)
