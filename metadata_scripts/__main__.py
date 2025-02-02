import copy
import json
import logging
import re
import time
from datetime import datetime

import pandas as pd
from translate import Translator

from metadata_scripts.all_texts_reader import process_all_texts_data
from metadata_scripts.demography_reader import process_demography_data
from metadata_scripts.habits_reader import process_habits_data
from metadata_scripts.helpers import format_data
from metadata_scripts.icd9_reader import process_icd9_data
from metadata_scripts.icdo_reader import process_icdo_data
from metadata_scripts.id_reader import generate_patient_index
from metadata_scripts.medicines_reader import process_medicines_data
from metadata_scripts.openai_api import translate_text_llm
from metadata_scripts.pathologic_report_reader import process_pathology_data

logging.basicConfig(
    filename="errors.log",  # File where logs are stored
    level=logging.DEBUG,  # Log everything from DEBUG level and above
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format of the logs
)


def extract_free_text_and_json_from_llm_analysis(llm_analysis):
    """
    Extract free text and JSON part from the COMBINED_ANALYSIS field in the pathology analysis.
    """
    free_text = ''
    json_data = {}

    try:
        combined_analysis = llm_analysis.get('COMBINED_ANALYSIS', '')
        if not isinstance(combined_analysis, str):
            logging.error("COMBINED_ANALYSIS is not a string.")
            return free_text, json_data

        free_text = re.split(r'```json', combined_analysis)[0].strip()
        json_part_match = re.search(r'```json\s*(\{.*?})\s*```', combined_analysis, re.DOTALL)

        if json_part_match:
            json_part = json_part_match.group(1).strip()
            try:
                json_data = json.loads(json_part)
            except json.JSONDecodeError:
                logging.error("Failed to decode JSON. Please check the formatting.")
        else:
            logging.error("No JSON section found in the combined analysis.")
    except Exception as e:
        logging.error(f"An error occurred during extraction: {e}")

    return free_text, json_data


def translate_gender(hebrew_text):
    """
    Translates the Hebrew words for gender ('זכר' for male, 'נקבה' for female) into English.

    Args:
        hebrew_text (str): The Hebrew text to translate.

    Returns:
        str: The translated text ('male' or 'female'). If the input doesn't match, return 'Unknown'.
    """
    # Dictionary mapping Hebrew terms to English
    translation_map = {
        'זכר': 'Male',
        'נקבה': 'Female'
    }

    # Return the translation or 'Unknown' if the input doesn't match
    return translation_map.get(hebrew_text, 'Unknown')


def translate_demography(patient_data, medical_texts_json_data):
    """
    Translates the nationality, religion/ethnicity, and country of birth fields from Hebrew to English.
    """
    try:
        translator = Translator(from_lang="hebrew", to_lang="english")
        nationality = translator.translate(patient_data['demography'].get('NATIONALITY_NAME', ''))
        religion = translator.translate(patient_data['demography'].get('RELIGION_NAME', ''))
        country_of_birth = translator.translate(patient_data['demography'].get('COUNTRYBIRTH_NAME', ''))
        ethnicity = medical_texts_json_data.get('ethnicity', '')

        # Check if ethnicity is valid (not None and not empty)
        if ethnicity and ethnicity.lower() != 'none':
            religion_ethnicity = f"Religion/Ethnicity: {religion}/{ethnicity}"
        else:
            religion_ethnicity = f"Religion: {religion}"

        return (f"Nationality: {nationality} \n"
                f"{religion_ethnicity}\nCOB: {country_of_birth}")
    except Exception as e:
        logging.error(f"Error in translating demography: {e}")
        return "Translation Error"


def format_habits(habits_data):
    """
    Combines all habits into a single string, translates, and formats the result.

    Args:
        habits_data (dict): Dictionary containing habit information.

    Returns:
        str: Formatted and translated string of habits with each entry on a new line, or 'None' if no data is available.
    """
    if not habits_data:
        return "None"

    # Combine all entries as "key: value" and join them with commas
    combined_habits = '\n'.join([f"{k}: {v}" for k, v in habits_data.items() if v])

    print(combined_habits)

    # Translate and clean the combined entry
    translated_formatted_habits = translate_and_clean_habits(combined_habits)

    return translated_formatted_habits if translated_formatted_habits else "None"


def clean_habits_text(text):
    # Define a list of specific bracketed phrases to remove
    phrases_to_remove = ["(detail)", "(Detail)", "(indicate quantity)", "(Indicate quantity)", "(specify quantity)",
                         "(Specify quantity)"]

    # Replace each phrase in the list with an empty string
    for phrase in phrases_to_remove:
        text = text.replace(phrase, "")

    # Clean up extra spaces after removals
    cleaned_text = ' '.join(text.split())
    return cleaned_text


def translate_and_clean_habits(text):
    """
    Translate the text only if existing translation is 'None'.

    Args:
        text (str): The text to be translated.

    Returns:
        str: The translated text.
    """

    cleaned_translation = "None"

    # If existing translation is None or empty, call GPT-4 API for translation
    if not text.strip() == 'None':
        step_start = time.time()
        translation = translate_text_llm(text)
        print(f"Translating habits text took {time.time() - step_start:.2f} seconds", flush=True)
        # Clean the text from unnecessary text
        cleaned_translation = clean_habits_text(translation)

    # If there's an existing translation, return it
    return cleaned_translation


def format_family_history(family_history):
    """
    Formats family history into a readable string.
    """
    formatted_history = ""
    try:
        for side, relatives in family_history.items():
            side_output = f"{side.capitalize()}:\n"
            has_valid_data = False  # Flag to check if there's any valid data

            for relative_data in relatives:
                # Check if 'relative' and all other keys are 'None' or missing
                if not any(relative_data.values()):
                    side_output += " None\n\n"
                else:
                    # Extract and format each attribute if valid data is present
                    has_valid_data = True
                    relative = relative_data.get('relative', 'Unknown')
                    condition = relative_data.get('condition', 'Unknown')
                    age_of_onset = relative_data.get('age_of_onset', 'Unknown')
                    mutations = ', '.join(relative_data.get('mutations', ['Unknown'])) if relative_data.get(
                        'mutations') else 'Unknown'

                    # Embed the variables into the formatted string
                    side_output += (
                        f"  Relative: {relative}\n"
                        f"  Condition: {condition}\n"
                        f"  Age of onset: {age_of_onset}\n"
                        f"  Mutations: {mutations}\n\n"
                    )

            # Add the side section to formatted history, either detailed or as 'None'
            formatted_history += side_output if has_valid_data else f"{side.capitalize()}: None\n\n"

    except Exception as e:
        logging.error(f"Error in formatting family history: {e}")
    return formatted_history.strip()


def format_allergies(allergies):
    """
    Formats allergies into a readable string.
    """
    formatted_allergies = ""
    try:
        if allergies.get('drug_related'):
            formatted_allergies += "Drug-related Allergies:\n" + '\n'.join(
                [f"  - {allergy}" for allergy in allergies['drug_related']])
            formatted_allergies += "\n\n"
        else:
            formatted_allergies += "Drug-related Allergies: None\n\n"

        if allergies.get('non_drug_related'):
            formatted_allergies += "Non-drug-related Allergies:\n" + '\n'.join(
                [f"  - {allergy}" for allergy in allergies['non_drug_related']])
        else:
            formatted_allergies += "Non-drug-related Allergies: None\n"
    except Exception as e:
        logging.error(f"Error in formatting allergies: {e}")
    return formatted_allergies


def format_medicines_by_date(patient_data, sample_date, prior=True):
    """
    Extracts and formats medicines from patient data based on whether they were issued
    before or after a specified date.

    Args:
        patient_data (dict): Dictionary containing patient data with 'medicines' key.
        sample_date (str or datetime): The reference date for filtering medicines. Format as 'YYYY-MM-DD' if a string.
        prior (bool): If True, returns medicines issued before the sample_date.
                      If False, returns medicines issued on or after the sample_date.

    Returns:
        str: Formatted string listing the filtered medicines with their issue dates or 'None' if no data.
    """

    medicines = patient_data.get('medicines', [])

    # Filter medicines based on the prior/post criteria
    filtered_medicines = [
        f"{med['MEDICINE_NAME']}, {med['MEDICINE_DOSAGE']} ({med['ISSUED_ON_STRING']})"
        for med in medicines
        if 'ISSUED_ON_DATETIME' in med and (
                (prior and med['ISSUED_ON_DATETIME'] < sample_date) or
                (not prior and med['ISSUED_ON_DATETIME'] >= sample_date)
        )
    ]

    # Return the formatted string or 'None' if no data
    return ', '.join(filtered_medicines) if filtered_medicines else 'None'


def format_genetic_data(genetic_data):
    """
    Formats genetic data into a readable string.
    """
    formatted_genetic_data = ""
    try:
        if not genetic_data:
            return formatted_genetic_data
        else:
            for key, value in genetic_data.items():
                if isinstance(value, list):
                    value = ', '.join(map(str, value))
                if not value:
                    value = "Not provided"
                formatted_genetic_data += f"{key}: {value}\n"
    except Exception as e:
        logging.error(f"Error in formatting genetic data: {e}")
    return formatted_genetic_data.strip()


def combine_tumor_stage_or_grade(pathology_value, medical_value):
    """
    Combines tumor stage or grade from pathology and medical texts data.
    If pathology data is 'None', replace it with an empty string.
    If medical data is 'None', return 'None' for that value.

    Args:
        pathology_value (str): Pathology JSON data.
        medical_value (str): Medical texts JSON data.

    Returns:
        str: Concatenated tumor stage/grade, or 'None' if medical_data is None.
    """

    # Replace 'None' in pathology_value with an empty string
    if pathology_value == 'None' or not pathology_value:
        pathology_value = ''

    # If medical_value is 'None', return 'None'
    if medical_value == 'None':
        return 'None'

    pathology_value = format_data(pathology_value)
    medical_value = format_data(medical_value)

    # Combine the pathology and medical values and return
    return f"{pathology_value} {medical_value}".strip()


def format_tissue_data(tissue_data):
    """
    Processes a list of dictionaries with tissue site information and details,
    generating a formatted string with line breaks.

    Args:
        tissue_data (list): A list of dictionaries with 'site' and 'details' keys.

    Returns:
        str: Formatted string with site and details.
    """
    if tissue_data == 'None':
        return 'None'

    formatted_text = ""

    for tissue in tissue_data:
        site = tissue.get('site', 'Unknown')

        # Handle 'details' as a list or single string
        details = tissue.get('details', 'No details available')
        if isinstance(details, list):
            details_text = ', '.join(details)
        else:
            details_text = details

        # Add to formatted text
        formatted_text += f"Site: {site}\nDetails: {details_text}\n\n"

    return formatted_text.strip()  # Remove trailing newlines if any


def combine_general_cancer_type(pathology_cancer_type, medical_texts_cancer_type):
    """
    Retrieves the general cancer type, prioritizing data from pathology reports. If the pathology
    data is missing or 'None', it falls back to using the data from medical texts.

    Args:
        pathology_cancer_type (str): The general cancer type from pathology reports.
        medical_texts_cancer_type (str): The general cancer type from medical texts.

    Returns:
        str: The general cancer type, or 'None' if no data is available.
    """

    # Return the pathology cancer type if available, otherwise use medical texts
    return pathology_cancer_type if pathology_cancer_type != 'None' \
        else medical_texts_cancer_type if medical_texts_cancer_type != 'None' else 'None'


def combine_genetic_metadata(pathology_metadata, medical_texts_metadata):
    """
    Combines genetic metadata from pathology and medical texts. If only one source has data,
    returns that data; if both are present, they are concatenated with labels. If both sources
    are missing or 'None', returns 'None'.

    Args:
        pathology_metadata (dict or str): Genetic metadata from pathology reports, which may be a dictionary
            or formatted string representation.
        medical_texts_metadata (dict or str): Genetic metadata from medical texts, which may be a dictionary
            or formatted string representation.

    Returns:
        str: A combined string of genetic metadata labeled by source or 'None' if no data is available.
    """

    # Format the metadata from both sources
    pathology_metadata = format_data(pathology_metadata)
    medical_texts_metadata = format_data(medical_texts_metadata)

    # If pathology data is 'None', use medical texts data if available
    if pathology_metadata == 'None':
        combined_metadata = medical_texts_metadata
    # If both sources have data, concatenate with labels for each source
    elif pathology_metadata != 'None' and medical_texts_metadata != 'None':
        combined_metadata = (
            f"Pathology report: {pathology_metadata}\n\n"
            f"Medical texts: {medical_texts_metadata}"
        ).strip()
    else:
        combined_metadata = pathology_metadata  # If only pathology data is available

    # Return combined metadata or 'None' if all data is missing or empty
    return combined_metadata if combined_metadata else 'None'


def combine_metastases_data(pathology_metastases_data, medical_texts_metastases_data):
    """
    Combines metastases data from pathology reports and medical texts. If only one source has data,
    returns that data; if both are present, they are concatenated with labels. If both sources
    are missing or 'None', returns 'None'.

    Args:
        pathology_metastases_data (dict or str): The metastases data from pathology reports, which may be a dictionary
            or formatted string representation.
        medical_texts_metastases_data (dict or str): The metastases data from medical texts, which may be a dictionary
            or formatted string representation.

    Returns:
        str: A combined string of metastases data labeled by source or 'None' if no data is available.
    """

    # Format the metastases data from both sources
    pathology_metastases_data = format_data(pathology_metastases_data)
    medical_texts_metastases_data = format_data(medical_texts_metastases_data)

    # If pathology data is 'None', use medical texts data if available
    if pathology_metastases_data == 'None':
        combined_metastases = medical_texts_metastases_data
    # If both sources have data, concatenate with labels for each source
    elif pathology_metastases_data != 'None' and medical_texts_metastases_data != 'None':
        combined_metastases = (
            f"Pathology report: {pathology_metastases_data}\n\n"
            f"Medical texts: {medical_texts_metastases_data}"
        ).strip()
    else:
        combined_metastases = pathology_metastases_data  # If only pathology data is available

    # Return combined metastases data or 'None' if all data is missing or empty
    return combined_metastases if combined_metastases else 'None'


def combine_ptnm_data(pathology_ptnm_data, medical_texts_ptnm_data):
    """
    Combines pTNM data from pathology reports and medical texts. If only one source has data,
    returns that data; if both are present, they are concatenated with labels. If both sources
    are missing or 'None', returns 'None'.

    Args:
        pathology_ptnm_data (dict or str): The pTNM data from pathology reports, which may be a dictionary
            or formatted string representation.
        medical_texts_ptnm_data (dict or str): The pTNM data from medical texts, which may be a dictionary
            or formatted string representation.

    Returns:
        str: A combined string of pTNM data labeled by source or 'None' if no data is available.
    """

    combined_ptnm_data = None  # Initialize combined data as None

    # Format the pTNM data from pathology and medical texts using format_data function
    pathology_ptnm_data = format_data(pathology_ptnm_data)
    medical_texts_ptnm_data = format_data(medical_texts_ptnm_data)

    # If pathology data is 'None', use medical texts data if available
    if pathology_ptnm_data == 'None':
        combined_ptnm_data = medical_texts_ptnm_data
    # If both sources have data, concatenate with labels for each source
    elif pathology_ptnm_data != 'None' and medical_texts_ptnm_data != 'None':
        combined_ptnm_data = (
            f"Pathology report: {pathology_ptnm_data}\n\n"
            f"Medical texts: {medical_texts_ptnm_data}"
        ).strip()

    # Return combined pTNM data or 'None' if all data is missing or empty
    return combined_ptnm_data if combined_ptnm_data else 'None'


def save_patient_metadata_to_excel(patient_index, patients_data, pathology_analysis, medical_texts_analysis,
                                   file_path):
    """
    Saves patient data into an Excel file, incorporating both patient metadata and
    analysis from GPT (both human-readable and JSON format).
    If data for any column is missing, the cell is left empty.

    Args:
        patient_index (dict): The dictionary containing patient identifiers and metadata.
        patients_data (dict): The dictionary containing basic patient information.
        pathology_analysis (dict): The LLM analysis, containing both the human-readable summary and JSON data.
        medical_texts_analysis (dict): The LLM analysis, containing both the human-readable summary and JSON data.
        file_path (str): The name of the file to save the metadata (default is 'patient_metadata.xlsx').
    """

    # Define the columns as per your requirement
    columns = [
        'OC', 'TYPE OF CANCER', 'YEAR OF BIRTH', 'DATE OF DEATH', 'SEX', 'DEMOGRAPHY',
        'DATE OF PROCEDURE/SAMPLE ARRIVAL',
        'FAMILY HISTORY', 'PAST PATHOLOGIES ', 'ALLERGIES', 'HABITS', 'TYPE OF PROCEDURE', 'DIAGNOSIS',
        'METASTASES', 'TUMOR STAGE/GRADE', 'PTNM/PTMN', 'TREATMENTS PREVIOUS TO SAMPLE ARRIVAL',
        'TREATMENTS POST TO SAMPLE ARRIVAL', 'STAINED FOR', 'IMMUNOSTAINING RESULTS',
        'IMMUNISTAININGS FOR MMR', 'TISSUES EXAMINED/STAINED', 'PDL1', 'TUMOR BURDEN TMB', '% CANCER CELLS', 'CEA',
        'METADATA', 'COMMENTS', 'LLM OUTPUT'
    ]

    # List to hold all rows of data
    data_rows = []
    logging.info("Starting the process of saving patient metadata to Excel.")

    # Iterate through each patient key in the patient index
    for patient_id in patient_index.keys():
        # Log the current patient ID and related index information
        logging.info(f"Processing patient_id: {patient_id}, OC: {patient_index[patient_id]['OC']}")

        # Extract patient data
        patient_data = patients_data.get(patient_id, {})
        if patient_data:
            logging.info(f"Patient data found for ID_BAZNAT {patient_id}")
        else:
            logging.warning(f"WARNING: No patient data found for ID_BAZNAT {patient_id}")

        # Extract pathology analysis data
        pathology_analysis_data = pathology_analysis.get(patient_id, None)
        if pathology_analysis_data:
            logging.info(f"Pathology analysis data found for ID_BAZNAT {patient_id}")
        else:
            logging.warning(f"WARNING: No pathology analysis data found for ID_BAZNAT {patient_id}")

        # Extract medical texts analysis data
        medical_texts_analysis_data = medical_texts_analysis.get(patient_id, None)
        if medical_texts_analysis_data:
            logging.info(f"Medical texts analysis data found for ID_BAZNAT {patient_id}")
        else:
            logging.warning(f"WARNING: No medical texts analysis data found for ID_BAZNAT {patient_id}")

        if not pathology_analysis_data or not medical_texts_analysis_data:
            continue

        # Extract summaries and JSON data for pathology and medical text analysis
        pathology_free_text_summary, pathology_json_data = extract_free_text_and_json_from_llm_analysis(
            pathology_analysis_data)
        medical_texts_free_text_summary, medical_texts_json_data = extract_free_text_and_json_from_llm_analysis(
            medical_texts_analysis_data)

        # Extract patient data for the specified columns
        row = {
            'OC': patient_data.get('OC', ''),
            'TYPE OF CANCER': pathology_json_data.get('general_cancer_type', 'None'),
            'YEAR OF BIRTH': patient_data['demography']['BIRTHDATE'][:4] if 'demography' in patient_data else '',
            'DATE OF DEATH': patient_data['demography'].get('DEATHDATE') if patient_data['demography'].get(
                'DEATHDATE') else 'None',
            'SEX': translate_gender(patient_data['demography'].get('GENDER_NAME', '')),
            'DEMOGRAPHY': translate_demography(patient_data, medical_texts_json_data),
            'DATE OF PROCEDURE/SAMPLE ARRIVAL': patient_data.get('PROCEDURE_DATE', ''),
            'FAMILY HISTORY': format_family_history(medical_texts_json_data.get('family_history')),
            'PAST PATHOLOGIES ': ', '.join([f"{item['PATHOLOGY']} ({item['FIRST_DIAGNOSE_DATE']})" for item in
                                            patient_data.get('icd9', [])]) if patient_data.get('icd9') else 'None',
            'ALLERGIES': format_allergies(medical_texts_json_data.get('allergies')),
            'HABITS': format_habits(patient_data.get('habits', {})),
            'COMMENT': '',
            'TYPE OF PROCEDURE': '. '.join([str(procedure).strip('.') for procedure in
                                            pathology_json_data.get('type_of_procedure_performed',
                                                                    [])]) if pathology_json_data.get(
                'type_of_procedure_performed') else 'None',
            'DIAGNOSIS': format_data(pathology_json_data.get('diagnosis_or_type_of_cancer', 'None')),
            'METASTASES': format_data(pathology_json_data.get('metastases', 'None')),
            'TUMOR STAGE/GRADE': format_data(pathology_json_data.get('tumor_stage_or_grade', 'None')),
            'PTNM/PTMN': format_data(pathology_json_data.get('ptnm_results', 'None')),
            'TREATMENTS PREVIOUS TO SAMPLE ARRIVAL': format_medicines_by_date(patient_data,
                                                                              patient_data.get('PROCEDURE_DATE'),
                                                                              prior=True),
            'TREATMENTS POST TO SAMPLE ARRIVAL': format_medicines_by_date(patient_data,
                                                                          patient_data.get('PROCEDURE_DATE'),
                                                                          prior=False),
            'STAINED FOR': ', '.join([str(marker) for marker in
                                      pathology_json_data.get('tissues_stained_for', [])]) if pathology_json_data.get(
                'tissues_stained_for') else 'None',
            'IMMUNOSTAINING RESULTS': format_data(pathology_json_data.get('immunostaining_results', 'None')),
            'IMMUNISTAININGS FOR MMR': str(pathology_json_data.get('immunostaining_for_MMR', 'None')),
            'TISSUES EXAMINED/STAINED': format_tissue_data(
                pathology_json_data.get('tissues_examined_and_stained', 'None')),
            'PDL1': str(pathology_json_data.get('PDL1', 'None')),
            'TUMOR BURDEN TMB': str(pathology_json_data.get('tumor_burden_TMB', 'None')),
            '% CANCER CELLS': str(pathology_json_data.get('percentage_of_cancer_cells_stained_for_PDL1', 'None')),
            'CEA': str(pathology_json_data.get('CEA', 'None')),
            'METADATA': combine_genetic_metadata(pathology_json_data.get('genetic_metadata', 'None'),
                                                 medical_texts_json_data.get('genetic_metadata', 'None')),
            'COMMENTS': pathology_json_data.get('comments', ''),
            'LLM OUTPUT': f"{pathology_free_text_summary} \n {str(pathology_json_data)} \n "
                          f"{medical_texts_free_text_summary} \n {str(medical_texts_json_data)}"
        }

        # Add the row to the list
        data_rows.append(row)

    # Create a DataFrame with all rows and the specified columns
    df = pd.DataFrame(data_rows, columns=columns)

    # Save the DataFrame to an Excel file
    df.to_excel(file_path, index=False)
    logging.info(f"Data saved to {file_path}")


def get_type_of_procedure(type_of_procedure_data):
    result = '. '.join([str(procedure).strip('.') for procedure in type_of_procedure_data])

    return result if result else 'None'


def save_other_cancer_data_to_excel(patient_index, patients_data, pathology_analysis, file_path):
    """
    Saves selected pathology and medicines data into an Excel file with specific columns.
    If data for any column is missing, the cell is left empty.

    Args:
        patient_index (dict): The dictionary containing patient identifiers and metadata.
        patients_data (dict): The dictionary containing basic patient information.
        pathology_analysis (dict): The LLM analysis, containing both the human-readable summary and JSON data.
        file_path (str): The name of the file to save the metadata.
    """

    # Define the columns to be included in the Excel file
    columns = [
        'OC', 'TYPE OF CANCER', 'DATE OF DIAGNOSIS', 'TYPE OF PROCEDURE', 'DIAGNOSIS',
        'METASTASES', 'TUMOR STAGE/GRADE', 'PTNM/PTMN', 'TREATMENTS PREVIOUS TO SAMPLE ARRIVAL',
        'TREATMENTS POST TO SAMPLE ARRIVAL', 'STAINED FOR', 'IMMUNOSTAINING RESULTS',
        'IMMUNISTAININGS FOR MMR', 'TISSUES EXAMINED/STAINED', 'PDL1', 'TUMOR BURDEN TMB', '% CANCER CELLS', 'CEA',
        'METADATA', 'COMMENTS', 'LLM OUTPUT'
    ]

    # List to hold all rows of data
    data_rows = []
    logging.info("Starting the process of saving pathology and medicines data to Excel.")

    # Iterate through each patient key in the patient index
    for patient_id in patient_index.keys():
        first_row_added = False

        # Extract patient data
        patient_data = patients_data.get(patient_id, {})
        pathology_analysis_data = pathology_analysis.get(patient_id, {})

        for lab_test_date, analysis_result in pathology_analysis_data.items():
            # Extract summaries and JSON data from pathology analysis
            pathology_free_text_summary, pathology_json_data = extract_free_text_and_json_from_llm_analysis(
                analysis_result
            )

            # Populate each row with the selected columns
            row = {
                'OC': patient_data.get('OC', '') if not first_row_added else '',
                'TYPE OF CANCER': pathology_json_data.get('general_cancer_type', 'None'),
                'DATE OF DIAGNOSIS': lab_test_date,
                'TYPE OF PROCEDURE': format_data(pathology_json_data.get('type_of_procedure_performed', [])),
                'DIAGNOSIS': format_data(pathology_json_data.get('diagnosis_or_type_of_cancer', 'None')),
                'METASTASES': format_data(pathology_json_data.get('metastases', 'None')),
                'TUMOR STAGE/GRADE': pathology_json_data.get('tumor_stage_or_grade', 'None'),
                'PTNM/PTMN': format_data(pathology_json_data.get('ptnm_results', 'None')),
                'TREATMENTS PREVIOUS TO SAMPLE ARRIVAL': format_medicines_by_date(patient_data,
                                                                                  patient_data.get('PROCEDURE_DATE'),
                                                                                  prior=True),
                'TREATMENTS POST TO SAMPLE ARRIVAL': format_medicines_by_date(patient_data,
                                                                              patient_data.get('PROCEDURE_DATE'),
                                                                              prior=False),
                'STAINED FOR': ', '.join(pathology_json_data.get('tissues_stained_for', [])),
                'IMMUNOSTAINING RESULTS': format_data(pathology_json_data.get('immunostaining_results', 'None')),
                'IMMUNISTAININGS FOR MMR': pathology_json_data.get('immunostaining_for_MMR', 'None'),
                'TISSUES EXAMINED/STAINED': format_tissue_data(
                    pathology_json_data.get('tissues_examined_and_stained', 'None')),
                'PDL1': pathology_json_data.get('PDL1', 'None'),
                'TUMOR BURDEN TMB': pathology_json_data.get('tumor_burden_TMB', 'None'),
                '% CANCER CELLS': pathology_json_data.get('percentage_of_cancer_cells_stained_for_PDL1', 'None'),
                'CEA': pathology_json_data.get('CEA', 'None'),
                'METADATA': format_data(pathology_json_data.get('genetic_metadata', 'None')),
                'COMMENTS': pathology_json_data.get('comments', ''),
                'LLM OUTPUT': f"{pathology_free_text_summary}\n{pathology_json_data}"
            }

            if not first_row_added:
                first_row_added = True

            # Append the row to the data rows
            data_rows.append(row)

    # Create DataFrame and save to Excel
    df = pd.DataFrame(data_rows, columns=columns)
    df.to_excel(file_path, index=False)
    logging.info(f"Data saved to {file_path}")


def main():
    logging.info("Beginning metadata script")
    # Start tracking the overall time
    overall_start = time.time()

    # Step 1: Generate the initial patient_index
    step_start = time.time()
    patient_index = generate_patient_index()
    print(f"Step 1: Generate patient index took {time.time() - step_start:.2f} seconds", flush=True)
    print(patient_index, flush=True)

    # Step 2: Create a deep copy of patient_index for further processing
    step_start = time.time()
    patient_index_copy = copy.deepcopy(patient_index)
    print(f"Step 2: Deep copy patient index took {time.time() - step_start:.2f} seconds", flush=True)

    # Step 3: Process habits data using the copy of patient_index
    step_start = time.time()
    patients_data = process_habits_data(patient_index_copy)
    print(f"Step 3: Process habits data took {time.time() - step_start:.2f} seconds", flush=True)

    # Step 4: Process demography data using the copy
    step_start = time.time()
    patients_data = process_demography_data(patient_index_copy)
    print(f"Step 4: Process demography data took {time.time() - step_start:.2f} seconds", flush=True)

    # Step 5: Process ICDO data using the copy
    step_start = time.time()
    patients_data = process_icdo_data(patient_index_copy)
    print(f"Step 5: Process ICDO data took {time.time() - step_start:.2f} seconds", flush=True)

    # Step 6: Example of processing ICD9 data using the copy
    step_start = time.time()
    patients_data = process_icd9_data(patient_index_copy)
    print(f"Step 6: Process ICD9 data took {time.time() - step_start:.2f} seconds", flush=True)

    # Step 7: Process medicines data using the copy
    step_start = time.time()
    patients_data = process_medicines_data(patient_index_copy)
    print(f"Step 7: Process medicines data took {time.time() - step_start:.2f} seconds", flush=True)

    # Step 8: Process pathology report analysis using the copy (API Call)
    step_start = time.time()
    pathology_report_analysis = process_pathology_data()
    print(f"Step 8: Process pathology report analysis took {time.time() - step_start:.2f} seconds", flush=True)

    # Step 9: Process medical texts and summaries related to the patient and the procedure (API Call)
    step_start = time.time()
    medical_text_analysis = process_all_texts_data()
    print(f"Step 9: Process medical texts took {time.time() - step_start:.2f} seconds", flush=True)

    # Step 11 - Process pathology data for any other cancers the patient might have had prior or post same collection
    step_start = time.time()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_patient_metadata_to_excel(patient_index, patients_data, pathology_report_analysis, medical_text_analysis,
                                   file_path=f"../results/patient_metadata {current_time}.xlsx")
    print(f"Save metadata to Excel took {time.time() - step_start:.2f} seconds", flush=True)

    # Step 11 - Process pathology data for any other cancers the patient might have had prior or post same collection
    step_start = time.time()
    other_cancers_pathology = process_pathology_data(True)
    print(f"Step 10: Other cancers processing via pathology reports analysis took {time.time() - step_start:.2f}"
          f" seconds", flush=True)

    # Step 12 - Save other cancers data to a separate Excel sheet
    save_other_cancer_data_to_excel(patient_index, patients_data, other_cancers_pathology,
                                    file_path=f"../results/patient_metadata {current_time} other_cancers.xlsx")
    print(f"Save other cancers data to Excel took {time.time() - step_start:.2f} seconds", flush=True)

    # Print the total time taken
    print(f"Total time: {time.time() - overall_start:.2f} seconds", flush=True)
    logging.info("finished metadata script")


if __name__ == "__main__":
    main()
