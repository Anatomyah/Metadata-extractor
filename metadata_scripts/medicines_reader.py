import pandas as pd

from metadata_scripts.helpers import get_file_path


def load_and_prepare_medicines_data(file_path):
    """Load the Excel file and prepare the data by converting dates."""
    # Load the Excel file without assuming any column names or index columns
    df = pd.read_excel(file_path, dtype={'ID_BAZNAT': str}, header=0)

    # Create a new column with the date string
    df['A_DATE_STRING'] = df['A_DATE'].apply(
        lambda x: x.strftime('%d-%m-%Y') if pd.notna(x) else None)

    # Convert all timestamps to date only
    df['A_DATE'] = df['A_DATE'].dt.date

    return df


def process_medicines_data(results_dict):
    file_path = get_file_path('medicines')
    df = load_and_prepare_medicines_data(file_path)

    # Group the DataFrame by 'ID_BAZNAT'
    grouped = df.groupby('ID_BAZNAT')

    for id_baznat, group in grouped:
        # Sort the group by date to ensure chronological order
        sorted_group = group.sort_values(by='A_DATE')

        medicines_data = []

        # Iterate over each row in the sorted group
        for index, row in sorted_group.iterrows():
            medicines_data.append({
                'ISSUED_ON_STRING': row['A_DATE_STRING'],
                'ISSUED_ON_DATETIME': row['A_DATE'],
                'MEDICINE_NAME': row['M_DESCRIPTION'],
                'MEDICINE_DOSAGE': row['M_DOSAGE']
            })

        if id_baznat in results_dict:
            results_dict[id_baznat]['medicines'] = medicines_data
        else:
            # If the ID_BAZNAT does not exist in results_dict, you can add it or handle it differently
            results_dict[id_baznat] = {'medicines': medicines_data}

    return results_dict
