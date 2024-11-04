import pandas as pd

from metadata_scripts.helpers import get_file_path


def load_and_prepare_icd9_data(file_path):
    """Load the Excel file and prepare the data by converting dates."""
    # Load the Excel file without assuming any column names or index columns
    df = pd.read_excel(file_path, dtype={'ID_BAZNAT': str}, header=0)

    df['FIRST_DIAGNOSE_DATE'] = df['FIRST_DIAGNOSE_DATE'].apply(
        lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)

    return df


def process_icd9_data(results_dict):
    file_path = file_path = get_file_path('icd9')
    df = load_and_prepare_icd9_data(file_path)

    # Group the DataFrame by 'ID_BAZNAT'
    grouped = df.groupby('ID_BAZNAT')

    for id_baznat, group in grouped:
        # Sort the group by date to ensure chronological order
        sorted_group = group.sort_values(by='FIRST_DIAGNOSE_DATE')

        icd9_data = []

        # Iterate over each row in the sorted group
        for index, row in sorted_group.iterrows():
            icd9_data.append({
                'PATHOLOGY': row['CODE_TEXT'],
                'FIRST_DIAGNOSE_DATE': row['FIRST_DIAGNOSE_DATE'],
            })

        if id_baznat in results_dict:
            results_dict[id_baznat]['icd9'] = icd9_data
        else:
            # If the ID_BAZNAT does not exist in results_dict, you can add it or handle it differently
            results_dict[id_baznat] = {'icd9': icd9_data}

    return results_dict
