import pandas as pd

from metadata_scripts.helpers import get_file_path, clean_keys


def get_unique_paragraph_names(data_frame):
    # Extract the 'PARAGRAPH_NAME' column and get unique values
    unique_paragraph_names = data_frame['PARAGRAPH_NAME'].unique()

    # Convert to a list
    return unique_paragraph_names.tolist()


def get_unique_section_names_per_paragraph_name(data_frame):
    """Extract unique SECTION_NAME values for each PARAGRAPH_NAME in the DataFrame."""
    unique_sections = {}

    for paragraph_name in data_frame['PARAGRAPH_NAME'].unique():
        # Filter rows for the current paragraph name
        paragraph_rows = data_frame[data_frame['PARAGRAPH_NAME'] == paragraph_name]

        # Get unique SECTION_NAME values
        unique_section_names = paragraph_rows['SECTION_NAME'].unique().tolist()

        # Store the unique SECTION_NAME values
        unique_sections[paragraph_name] = unique_section_names

    return unique_sections


def load_and_prepare_habits_data(file_path):
    """Load the Excel file and prepare the data by converting dates."""
    # Load the Excel file without assuming any column names or index columns
    df = pd.read_excel(file_path, dtype={'ID_BAZNAT': str}, header=0)

    # Convert the 'A_DATE' column to datetime, handling both date formats
    df['A_DATE'] = pd.to_datetime(df['A_DATE'], format='%d/%m/%Y %H:%M:%S', dayfirst=True, errors='coerce')
    df['A_DATE'] = df['A_DATE'].fillna(pd.to_datetime(df['A_DATE'], format='%d/%m/%Y', dayfirst=True, errors='coerce'))

    # Convert the 'A_DATE' column to a date-only format
    df['A_DATE'] = df['A_DATE'].dt.date

    return df


def sort_data_by_date(df):
    """Sort the DataFrame by 'A_DATE'."""
    return df.sort_values(by='A_DATE')


def process_habits_data(results_dict):
    """Main function to process the data and extract the required information."""
    file_path = get_file_path('habits')
    df = load_and_prepare_habits_data(file_path)

    # Group the DataFrame by 'ID_BAZNAT'
    grouped = df.groupby('ID_BAZNAT')

    for id_baznat, group in grouped:
        # Sort the group by date to ensure chronological order
        sorted_group = group.sort_values(by='A_DATE')

        # Initialize a dictionary to hold the paragraph data for this ID_BAZNAT
        paragraph_data = {}

        # Loop over each paragraph name
        for paragraph_name in get_unique_paragraph_names(df):
            # Filter the rows for this paragraph name
            paragraph_rows = sorted_group[sorted_group['PARAGRAPH_NAME'] == paragraph_name]

            if not paragraph_rows.empty:
                # Get the most recent row for this paragraph name
                latest_row = paragraph_rows.iloc[-1]

                if pd.notna(latest_row['COMMENTS']):
                    paragraph_data[paragraph_name] = f"{latest_row['SECTION_NAME']}, {latest_row['COMMENTS']}"
                else:
                    paragraph_data[paragraph_name] = f"{latest_row['SECTION_NAME']}"
            else:
                # If there is no data for this paragraph name, handle by setting to None
                paragraph_data[paragraph_name] = None

        # Add the paragraph data to the results dictionary under the corresponding ID_BAZNAT
        cleaned_paragraph_data = clean_keys(paragraph_data)
        results_dict[id_baznat]['habits'] = cleaned_paragraph_data

    return results_dict
