import pandas as pd

from metadata_scripts.helpers import get_file_path


def load_and_prepare_icdo_data(file_path):
    """Load the Excel file and prepare the data by converting dates."""
    # Load the Excel file without assuming any column names or index columns
    df = pd.read_excel(file_path, dtype={'ID_BAZNAT': str}, header=0)

    df['FIRST_DIAGOSE_DATE'] = df['FIRST_DIAGOSE_DATE'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)

    return df


def process_icdo_data(results_dict):
    file_path = file_path = get_file_path('icdo')
    df = load_and_prepare_icdo_data(file_path)

    for index, row in df.iterrows():
        id_baznat = row['ID_BAZNAT']

        icdo_data = {
            'CANCER_NAME': row['NAME'],
            'FIRST_DIAGNOSE_DATE': row['FIRST_DIAGOSE_DATE'],
        }

        if id_baznat in results_dict:
            results_dict[id_baznat]['icdo'] = icdo_data
        else:
            # If the ID_BAZNAT does not exist in results_dict, you can add it or handle it differently
            results_dict[id_baznat] = {'icdo': icdo_data}

    return results_dict
