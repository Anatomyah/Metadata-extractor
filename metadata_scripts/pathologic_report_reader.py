import time

import pandas as pd

from metadata_scripts.helpers import clean_text_with_actual_newlines
from metadata_scripts.helpers import get_file_path
from metadata_scripts.openai_api import analyze_pathology_report, combine_batched_analyses


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

    return final_filtered_df


def clean_pathology_data(pathology_data_dict):
    """
    Clean all the text fields in the pathology data dictionary while preserving line breaks for readability.
    Also calculate the token count for the cleaned version with actual newlines.

    The function will clean each field by replacing unwanted characters and ensuring that the text retains
    its original line breaks. Additionally, it computes the token count for each cleaned entry using the
    actual newline format and prints the result.
    """
    for id_baznat, lab_test_entries in pathology_data_dict.items():
        for lab_test_date, entries in lab_test_entries.items():
            cleaned_entries = []

            # Iterate over the list of entries for each LAB_TEST_DATE
            for entry in entries:
                # Clean each field using the actual newline method
                cleaned_entry_newlines = {key: clean_text_with_actual_newlines(value) for key, value in entry.items()}

                # Add the cleaned entry (with newlines) to the list of cleaned entries for this date
                cleaned_entries.append(cleaned_entry_newlines)

            # Replace the original entries for this date with the cleaned entries
            pathology_data_dict[id_baznat][lab_test_date] = cleaned_entries

    return pathology_data_dict


def extract_pathology_data(filtered_df):
    """
    Extract pathology data and return a dictionary where ID_BAZNAT is the key,
    and LAB_TEST_DATE is used as the secondary key for each entry. If multiple
    rows share the same date, they will be stored as a list.
    """
    # Columns we are interested in
    relevant_columns = ['LAB_TEST_NAME', 'PATHOLOGY_BODY_PART_IN_SHORT',
                        'PATHOLOGY_BODY_PART_DETAILED', 'RESULT_DESC', 'RESULT_TEXT']

    # Initialize the dictionary to store data by ID_BAZNAT
    pathology_data_dict = {}

    # Iterate over each row in the filtered DataFrame
    for index, row in filtered_df.iterrows():
        id_baznat = row['ID_BAZNAT']
        lab_test_date = row['LAB_TEST_DATE']

        # Create an entry for the relevant columns
        entry = {column: row[column] for column in relevant_columns if pd.notna(row[column])}

        # If the ID_BAZNAT does not exist in the dictionary, initialize it
        if id_baznat not in pathology_data_dict:
            pathology_data_dict[id_baznat] = {}

        # If the LAB_TEST_DATE exists, append the entry to the list, otherwise create a new list
        if lab_test_date in pathology_data_dict[id_baznat]:
            pathology_data_dict[id_baznat][lab_test_date].append(entry)
        else:
            pathology_data_dict[id_baznat][lab_test_date] = [entry]

    # Clean the extracted data (clean the text within the dictionary)
    cleaned_data = clean_pathology_data(pathology_data_dict)

    return cleaned_data


def extract_combined_text_for_baznat(entries):
    """
    Extract and combine 'RESULT_DESC' and 'RESULT_TEXT' for a given set of entries.
    Returns the combined text formatted for better readability.
    """
    combined_entries = []

    for entry in entries:
        entry_str = ''
        result_desc = entry.get('RESULT_DESC', '')
        result_text = entry.get('RESULT_TEXT', '')

        # Format result description if it doesn't start with 'ראו'
        if result_desc:
            if not result_desc.startswith('ראו'):
                entry_str = f"**Result Description**: {result_desc}\n"

        # Handle result text, marking if it's too long
        if result_text:
            if result_text.startswith('התשובה ארוכה מידי'):
                entry_str += (
                    "\n***TEXT TOO LONG TO EXTRACT! MAKE SURE TO INCLUDE REMARK AT THE END OF THE ANALYSIS REPORT***\n"
                )
            else:
                entry_str += f"**Result Text**: {result_text}\n"

        # Append each structured entry to the combined list
        combined_entries.append(entry_str)

    # Return the list of structured entries formatted as bullet points
    readable_text = "\n".join(combined_entries)

    return readable_text


def generate_api_requests(pathology_data_dict, gpt_ready_text=None):
    """
    Generate API requests based on the combined text for each ID_BAZNAT.
    """
    api_requests = []

    for id_baznat, dates_data in pathology_data_dict.items():

        for lab_test_date, entries in dates_data.items():
            gpt_ready_text = extract_combined_text_for_baznat(entries)

        # If there's a valid, unified text, add to the API request list
        if gpt_ready_text:
            api_requests.append((id_baznat, gpt_ready_text))

    return api_requests


def generate_api_requests_by_date(pathology_data_dict):
    """
    Generate API requests based on individual test dates for each ID_BAZNAT.
    Each LAB_TEST_DATE will have its own analysis request.
    """
    api_requests = {}

    # Loop through each ID_BAZNAT in the pathology data dictionary
    for id_baznat, dates_data in pathology_data_dict.items():
        api_requests[id_baznat] = {}

        # Loop through each date's entries
        for lab_test_date, entries in dates_data.items():
            # Combine entries for this specific date
            gpt_ready_text = extract_combined_text_for_baznat(entries)

            # Only add the request if the combined text is not empty
            if gpt_ready_text:
                # Store each request under the ID_BAZNAT with the LAB_TEST_DATE as a key
                api_requests[id_baznat][lab_test_date] = gpt_ready_text

    return api_requests


def analyze_requests_by_date(api_requests):
    """
    Analyze each date's request individually by sending it to the LLM.
    Results are stored separately for each LAB_TEST_DATE within each ID_BAZNAT.
    """
    combined_results_by_baznat = {}

    # Iterate over each ID_BAZNAT and its associated dates
    for id_baznat, dates_data in api_requests.items():
        combined_results_by_baznat[id_baznat] = {}

        # Analyze each request per date
        for lab_test_date, cleaned_text in dates_data.items():
            request_start = time.time()

            # Ensure the lab_test_date is initialized as a dictionary within each ID_BAZNAT
            combined_results_by_baznat[id_baznat][lab_test_date] = {}

            # Analyze each date's request individually
            analysis = analyze_pathology_report(cleaned_text)

            # Store the analysis result in the dictionary under the specific date
            combined_results_by_baznat[id_baznat][lab_test_date]['COMBINED_ANALYSIS'] = analysis

            print(
                f"Pathology analysis for {id_baznat} on {lab_test_date} took {time.time() - request_start:.2f} seconds")

    return combined_results_by_baznat


def analyze_requests(api_requests):
    """
    Analyze all requests in the list by sending the text to the LLM.
    Handles batched requests by accumulating results for each ID_BAZNAT and combines them
    into a single coherent analysis if multiple batches are present.
    """
    combined_results_by_baznat = {}
    batch_counts = {}

    # Iterate over all API requests
    for iteration_count, (id_baznat, cleaned_text) in enumerate(api_requests):
        request_start = time.time()

        # Analyze each request by sending it to the OpenAI API
        analysis = analyze_pathology_report(cleaned_text)

        # Identify base ID_BAZNAT for batched requests
        base_id_baznat = id_baznat.split('_batch')[0] if '_batch' in id_baznat else id_baznat

        # Track batch count for each ID_BAZNAT
        batch_counts[base_id_baznat] = batch_counts.get(base_id_baznat, 0) + 1

        # Initialize entry if this is the first batch for the ID_BAZNAT
        if base_id_baznat not in combined_results_by_baznat:
            combined_results_by_baznat[base_id_baznat] = {
                'batches': []  # Store each batch analysis for later combination
            }

        # Append analysis result to the list of batches
        combined_results_by_baznat[base_id_baznat]['batches'].append(analysis)

        print(f"Pathology analysis for {id_baznat} ID_BAZNAT took {time.time() - request_start:.2f} seconds")

    # Combine batched analyses for each ID_BAZNAT with multiple batches
    for base_id_baznat, result_data in combined_results_by_baznat.items():
        if batch_counts[base_id_baznat] > 1:
            # Combine all batches for this ID_BAZNAT using the new function
            combined_analysis = combine_batched_analyses(result_data['batches'])
            print(f"\nCombined analysis for {base_id_baznat}:\n{combined_analysis}")
            combined_results_by_baznat[base_id_baznat]['COMBINED_ANALYSIS'] = combined_analysis
        else:
            # If only one batch, no need to combine
            combined_results_by_baznat[base_id_baznat]['COMBINED_ANALYSIS'] = result_data['batches'][0]

    return combined_results_by_baznat


def run_pathology_llm_analysis(pathology_data_dict, other_cancers_analysis):
    """
    Run the OpenAI analysis on a combined 'RESULT_DESC' and 'RESULT_TEXT' for each ID_BAZNAT,
    and store the results in the dictionary.
    This function processes arrays of dictionaries under each LAB_TEST_DATE for every ID_BAZNAT.
    """

    # Generate the API requests based on the pathology data
    if other_cancers_analysis:
        api_requests = generate_api_requests_by_date(pathology_data_dict)
        combined_results = analyze_requests_by_date(api_requests)
    else:
        api_requests = generate_api_requests(pathology_data_dict)
        combined_results = analyze_requests(api_requests)

    return combined_results


def process_pathology_data(other_cancers_analysis=False):
    """
    Load, clean, and restructure the pathology data.
    """
    # Load and prepare the data
    data = load_and_prepare_pathology_data('pathologic_report', 'ids', other_cancers_analysis)

    # Clean and extract pathology data
    cleaned_data = extract_pathology_data(data)
    # Prepare and send the pathology data to chatGPT for analysis
    final_results = run_pathology_llm_analysis(cleaned_data, other_cancers_analysis)
    return final_results
