import time

import pandas as pd

from metadata_scripts.helpers import get_file_path, clean_text_with_actual_newlines, fix_invalid_dates, \
    split_text_into_batches, count_tokens
from metadata_scripts.openai_api import analyze_all_texts_report, combine_batched_analyses


def load_and_prepare_all_texts(summaries_file_path, ids_file_path):
    """
    Load the doctor summaries and IDs files, prepare the data by merging based on ID_BAZNAT,
    and filter rows to keep only those where A_DATE is within 3 months prior to PROCEDURE_DATE
    and up to 3 months after PROCEDURE_DATE. Additionally, filter out rows where the COMMENTS
    are 'סיכום אחות מתאמת' or 'עו"ס הערכה ביופסיכוסוציאלית'.
    """
    summaries_df = pd.read_excel(get_file_path(summaries_file_path), dtype={'ID_BAZNAT': str}, header=0)

    # Load IDs file with procedure dates
    ids_df = pd.read_excel(get_file_path(ids_file_path), dtype={'ID_BAZNAT': str}, header=0)

    # Step 1: Clean PROCEDURE_DATE column
    # Trim extra spaces and ensure consistency in delimiters
    ids_df['PROCEDURE_DATE'] = ids_df['PROCEDURE_DATE'].str.strip().str.replace(r'[^\d.]', '.', regex=True)

    # Step 2: Identify invalid dates based on length (e.g., "31.10.22" is 8 characters long)
    invalid_dates = ids_df[~ids_df['PROCEDURE_DATE'].str.match(r'^\d{1,2}\.\d{1,2}\.\d{2}$', na=False)]
    if not invalid_dates.empty:
        print("Invalid PROCEDURE_DATE values before fixing:\n", invalid_dates)

    if not invalid_dates.empty:
        # Step 3: Manually fix invalid dates if any
        # Here, you can add custom rules to fix specific invalid date formats, e.g., incomplete years
        ids_df['PROCEDURE_DATE'] = ids_df['PROCEDURE_DATE'].apply(fix_invalid_dates)

    # Step 4: Convert PROCEDURE_DATE column to datetime after cleaning
    try:
        ids_df['PROCEDURE_DATE'] = pd.to_datetime(ids_df['PROCEDURE_DATE'], format='%d.%m.%y', errors='raise')
    except ValueError as e:
        print(f"Error during date conversion: {e}")

    # Step 5: Recheck for invalid dates after conversion
    invalid_dates_after_conversion = ids_df[ids_df['PROCEDURE_DATE'].isna()]
    if not invalid_dates_after_conversion.empty:
        print("Rows with invalid PROCEDURE_DATE after conversion:\n", invalid_dates_after_conversion)

    # Merge doctor summaries with IDs based on 'ID_BAZNAT'
    merged_df = pd.merge(summaries_df, ids_df, on='ID_BAZNAT', how='inner')

    # Define the range: 3 months before and 3 months after the PROCEDURE_DATE
    merged_df['min_date'] = merged_df['PROCEDURE_DATE'] - pd.DateOffset(months=3)
    merged_df['max_date'] = merged_df['PROCEDURE_DATE'] + pd.DateOffset(months=3)

    # Filter the data according to the defined range
    filtered_df = merged_df[(merged_df['A_DATE'] >= merged_df['min_date']) &
                            (merged_df['A_DATE'] <= merged_df['max_date'])]

    # Filter out rows where the COMMENTS are either 'סיכום אחות מתאמת' or 'עו"ס הערכה ביופסיכוסוציאלית'
    filtered_df = filtered_df[~filtered_df['COMMENTS'].isin(['סיכום אחות מתאמת', 'עו"ס הערכה ביופסיכוסוציאלית'])]

    # Drop the extra date columns used for filtering
    filtered_df = filtered_df.drop(columns=['min_date', 'max_date'])

    return filtered_df


def clean_all_texts_data(texts_df):
    """
    Clean the 'TEXT' column in the all_texts DataFrame while preserving line breaks for readability.
    Also, calculate the token count for each cleaned text if necessary (can be added).
    """
    # Apply the cleaning function to the 'TEXT' column
    texts_df['CLEANED_TEXT'] = texts_df['TEXT'].apply(clean_text_with_actual_newlines)

    return texts_df


def extract_and_prepare_all_texts(filtered_df):
    """
    Extract and clean the 'TEXT' column from the filtered DataFrame.
    Return a dictionary where ID_BAZNAT is the key and the cleaned TEXT is stored for each entry.
    """
    # Initialize the dictionary to store data by ID_BAZNAT
    all_texts_data_dict = {}

    # Iterate over each row in the filtered DataFrame
    for index, row in filtered_df.iterrows():
        id_baznat = row['ID_BAZNAT']

        # Cleaned text from the TEXT column
        cleaned_text = row['CLEANED_TEXT']

        # If the ID_BAZNAT does not exist in the dictionary, initialize it
        if id_baznat not in all_texts_data_dict:
            all_texts_data_dict[id_baznat] = []

        # Append the cleaned text entry to the list
        all_texts_data_dict[id_baznat].append(cleaned_text)

    return all_texts_data_dict


def generate_api_requests(all_texts_data_dict, max_tokens=120000):
    """
    Generate unified API requests based on the combined cleaned text for each ID_BAZNAT.
    Only split the text if it exceeds the max token limit.
    """
    api_requests = []

    # Iterate over each ID_BAZNAT and combine its corresponding texts
    for id_baznat, texts in all_texts_data_dict.items():
        # Combine all texts into a single string for this ID_BAZNAT
        combined_text = "\n\n".join(texts)

        if combined_text:  # Only proceed if the combined text is non-empty
            # Count the tokens of the entire combined text
            total_tokens = count_tokens(combined_text)

            # If the total token count exceeds the limit, split into batches
            if total_tokens > max_tokens:
                batches = split_text_into_batches(combined_text, max_tokens=max_tokens)

                # Create an API request for each batch
                for i, batch in enumerate(batches):
                    api_requests.append((f"{id_baznat}_batch_{i + 1}", batch))
            else:
                # If the token count is within the limit, send the entire text as a single request
                api_requests.append((id_baznat, combined_text))

    return api_requests


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
        analysis = analyze_all_texts_report(cleaned_text)

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

        print(f"Medical texts analysis for {id_baznat} took {time.time() - request_start:.2f} seconds")

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


def run_all_texts_llm_analysis(all_texts_data_dict):
    """
    Run the OpenAI analysis on the cleaned 'TEXT' for each ID_BAZNAT.
    This function processes arrays of cleaned text under each ID_BAZNAT.
    """

    # Generate the API requests based on the cleaned text data
    api_requests = generate_api_requests(all_texts_data_dict)

    # Run the OpenAI analysis on the generated requests
    combined_results = analyze_requests(api_requests)

    return combined_results


def process_all_texts_data(other_cancers_analysis=False, pathology_analysis=None):
    """
    Load, clean, and restructure the all_texts data.
    """
    # Load and prepare the data
    df = load_and_prepare_all_texts('all_texts', 'ids')

    # Clean the 'TEXT' column in the DataFrame
    cleaned_df = clean_all_texts_data(df)

    # Extract the cleaned text for further processing
    prepared_text_data = extract_and_prepare_all_texts(cleaned_df)
    # Run the analysis using OpenAI API
    final_results = run_all_texts_llm_analysis(prepared_text_data)

    return final_results
