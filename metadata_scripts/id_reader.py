import pandas as pd

from metadata_scripts.helpers import get_file_path


def clean_date_string(df):
    # Clean the 'PROCEDURE_DATE' column
    df['PROCEDURE_DATE'] = df['PROCEDURE_DATE'].astype(str)  # Ensure it's treated as a string
    df['PROCEDURE_DATE'] = df['PROCEDURE_DATE'].str.strip()  # Remove leading/trailing spaces
    df['PROCEDURE_DATE'] = df['PROCEDURE_DATE'].str.replace(r'[^0-9./-]', '', regex=True)  # Remove non-date characters

    return df


def generate_patient_index():
    file_path = get_file_path('ids')
    # Load the Excel file
    df = pd.read_excel(file_path, dtype={'ID_BAZNAT': str, 'PATIENT_ID': str})

    # Remove leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Drop any unnamed or empty columns
    df = df.loc[:, df.columns.notnull() & (df.columns != '')]

    # Clean the date string
    df = clean_date_string(df)

    # Ensure PROCEDURE_DATE is treated as datetime
    df['PROCEDURE_DATE'] = pd.to_datetime(df['PROCEDURE_DATE'], format='%d.%m.%y', dayfirst=True, errors='coerce')

    # Generate the dictionary with ID_BAZNAT as the key and a dictionary containing PROCEDURE_DATE as the value
    patient_dict = df.set_index('ID_BAZNAT').apply(
        lambda row: {'PROCEDURE_DATE': row['PROCEDURE_DATE'].date() if not pd.isna(row['PROCEDURE_DATE']) else None,
                     'OC': row['OC']},
        axis=1
    ).to_dict()

    return patient_dict
