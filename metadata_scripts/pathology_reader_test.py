import pandas as pd

from metadata_scripts.helpers import get_file_path


def load_and_prepare_pathology_data(pathology_file_path, ids_file_path, other_cancers_analysis):
    """
    Load the pathology report and IDs files, filter by date range to acquire OWNER_REFs,
    then filter the entire DataFrame by those OWNER_REFs.
    Return the filtered DataFrame.
    """

    # Load pathology report data
    pathology_df = pd.read_excel(get_file_path(pathology_file_path), dtype={'ID_BAZNAT': str}, header=0)

    # Load IDs file with procedure dates
    ids_df = pd.read_excel(get_file_path(ids_file_path), dtype={'ID_BAZNAT': str}, header=0)

    # Clean PROCEDURE_DATE column and convert to datetime
    ids_df['PROCEDURE_DATE'] = ids_df['PROCEDURE_DATE'].str.strip().str.replace(r'[^\d.]', '.', regex=True)
    ids_df['PROCEDURE_DATE'] = pd.to_datetime(ids_df['PROCEDURE_DATE'], format='%d.%m.%y', errors='coerce')

    # Convert LAB_TEST_DATE in pathology data to datetime
    pathology_df['LAB_TEST_DATE'] = pd.to_datetime(pathology_df['LAB_TEST_DATE'], errors='coerce')

    # Merge pathology data with IDs based on ID_BAZNAT
    merged_df = pd.merge(pathology_df, ids_df, on='ID_BAZNAT', how='inner')

    # Define date range: 1 day before to 5 days after the procedure
    merged_df['min_date'] = merged_df['PROCEDURE_DATE'] - pd.Timedelta(days=2)
    merged_df['max_date'] = merged_df['PROCEDURE_DATE'] + pd.Timedelta(days=2)

    # Step 1: Apply the date range filter to acquire OWNER_REFs within the date range
    if other_cancers_analysis:
        date_filtered_df = merged_df[(merged_df['LAB_TEST_DATE'] < merged_df['min_date']) |
                                     (merged_df['LAB_TEST_DATE'] > merged_df['max_date'])]
    else:
        date_filtered_df = merged_df[(merged_df['LAB_TEST_DATE'] >= merged_df['min_date']) &
                                     (merged_df['LAB_TEST_DATE'] <= merged_df['max_date'])]

    # Step 2: Get unique OWNER_REF values from the date-filtered DataFrame
    owner_refs_to_include = date_filtered_df['OWNER_REF'].unique()

    # Step 3: Filter the entire merged_df by these OWNER_REF values
    final_filtered_df = merged_df[merged_df['OWNER_REF'].isin(owner_refs_to_include)]

    # Print the final filtered data by OWNER_REF for each ID
    print("Final Filtered Data by OWNER_REF for each ID:")
    for index, row in final_filtered_df.iterrows():
        discrepancy = " (Discrepancy)" if row['PROCEDURE_DATE'].date() != row['LAB_TEST_DATE'].date() else ""
        print(f"{index + 1}. ID: {row['ID_BAZNAT']}, PROCEDURE_DATE: {row['PROCEDURE_DATE']}, "
              f"LAB_TEST_DATE: {row['LAB_TEST_DATE']}, OWNER_REF: {row['OWNER_REF']}{discrepancy}")

    return final_filtered_df


def process_pathology_data_test(other_cancers_analysis=False):
    """
    Load, clean, and restructure the pathology data.
    """

    # patient_index = generate_patient_index()

    # Load and prepare the data
    data = load_and_prepare_pathology_data('pathologic_report', 'ids', other_cancers_analysis)

# process_pathology_data_test()
