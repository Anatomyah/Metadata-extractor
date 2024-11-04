import pandas as pd

from metadata_scripts.helpers import get_file_path


def load_and_prepare_demography_data(file_path):
    """Load the Excel file and prepare the data by converting dates."""
    # Load the Excel file without assuming any column names or index columns
    df = pd.read_excel(file_path, dtype={'ID_BAZNAT': str}, header=0)

    # Convert the 'BIRTHDATE' column to a string in the 'YYYY-MM-DD' format
    df['BIRTHDATE'] = df['BIRTHDATE'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)
    df['DEATHDATE'] = df['DEATHDATE'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)

    return df


def process_demography_data(results_dict):
    file_path = get_file_path('demography')
    df = load_and_prepare_demography_data(file_path)

    # Iterate over each row in the demography DataFrame
    for index, row in df.iterrows():
        id_baznat = row['ID_BAZNAT']

        # Create a demography dictionary with relevant data
        demography_data = {
            'GENDER_NAME': row['GENDER_NAME'],
            'BIRTHDATE': row['BIRTHDATE'],
            'DEATHDATE': row['DEATHDATE'],
            'NATIONALITY_NAME': row['NATIONALITY_NAME'],
            'RELIGION_NAME': row['RELIGION_NAME'],
            'COUNTRYBIRTH_NAME': row['COUNTRYBIRTH_NAME'],
        }

        # Add the demography data to the corresponding ID_BAZNAT in results_dict
        if id_baznat in results_dict:
            results_dict[id_baznat]['demography'] = demography_data
        else:
            # If the ID_BAZNAT does not exist in results_dict, you can add it or handle it differently
            results_dict[id_baznat] = {'demography': demography_data}

    return results_dict
