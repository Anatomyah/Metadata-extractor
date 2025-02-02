import os

from openai import OpenAI

CLIENT = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)


def translate_text_llm(text):
    """
    Translate a given text using OpenAI's GPT-4 API.

    Args:
        text (str): The text to translate.

    Returns:
        str: Translated text.
    """
    # OpenAI API key
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Messages that guide the translation
    messages = [
        {
            "role": "system",
            "content": (
                "You are a translator. Translate the following text from Hebrew to English."
            ),
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    try:
        # Make the API call to OpenAI's GPT-4 model
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=500,  # Adjust this based on your expected output length
        )

        # Extract the response
        translated_text = response.choices[0].message.content.strip()
        return translated_text

    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None


def analyze_pathology_report(report_text):
    """
    Send the report text to OpenAI API for analysis to specifically extract relevant information.
    """

    # Instantiate the OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Constructing the message format for the OpenAI chat API
    messages = [
        {
            "role": "system",
            "content": (
                "You are analyzing a pathology report in Hebrew. Extract the following information and output it in "
                "the following format in English:\n"
                "1. pTNM results (also called pTMN).\n"
                "2. What were the tissues stained for (e.g., Ki-67, ER, PR, HER-2).\n"
                "3. Immunostaining results.\n"
                "4. Immunostaining for MMR.\n"
                "5. Describe what tissues were examined and stained.\n"
                "6. Percentage of cancer cells stained for PDL1.\n"
                "7. Tumor burden (TMB).\n"
                "8. CEA (Carcinoembryonic antigen).\n"
                "9. The tumor stage or grade (e.g., 4B).\n"
                "10. **General cancer type**: Match to one of these labels if applicable: Breast Cancer, Pancreatic Cancer, "
                "Colorectal Cancer, Lung Cancer, Ovarian Cancer, Endometrial Cancer. If none fit, respond as precisely as possible.\n"
                "11. The diagnosis and/or type of cancer (e.g., lobular carcinoma in situ, invasive carcinoma).\n"
                "12. The type of procedure performed.\n"
                "13. Metastases information.\n"
                "14. Any other relevant genetic data or metadata found in the pathology report (labeled 'genetic_metadata').\n"
                "15. **Comments**: If the raw text sent for analysis includes the exact phrase:\n"
                "\"\"\"\n***TEXT TOO LONG TO EXTRACT! MAKE SURE TO INCLUDE REMARK AT THE END OF THE ANALYSIS REPORT***\n\"\"\"\n"
                "then include the following:\n"
                "- In the 'Human-readable Summary,' add the phrase verbatim.\n"
                "- In the JSON response, add a 'comments' field with the message:\n"
                "  'pathology text too long to extract. Please extract manually'.\n"
                "If this phrase is not present in the raw text, do not include the 'comments' field in the JSON.\n\n"
                "For the human-readable summary, provide the extracted information in the same order as the list above, "
                "with each item labeled using the following format:\n"
                "1. **pTNM Results**: <extracted_data>.\n"
                "2. **Tissues Stained For**: <extracted_data>.\n"
                "...\n"
                "If any information is missing in the report, write 'Not provided in the report' instead.\n\n"
                "Please provide the output in two clearly separated sections:\n"
                "1. A human-readable summary (labeled 'Human-readable Summary'), including the phrase if present in the text.\n"
                "2. A **complete and valid JSON** response (labeled 'JSON Format'), with no truncation or incomplete parts. "
                "If the text includes the phrase, include a 'comments' field with the specific message.\n\n"
                "Use the following JSON structure, and ensure all keys are named using Python naming conventions "
                "(snake_case):\n"
                "\n"
                "```json\n"
                "{\n"
                "  \"ptnm_results\": null,\n"
                "  \"tissues_stained_for\": [],\n"
                "  \"immunostaining_results\": null,\n"
                "  \"immunostaining_for_mmr\": null,\n"
                "  \"tumor_burden_tmb\": null,\n"
                "  \"percentage_of_cancer_cells_stained_for_pdl1\": null,\n"
                "  \"cea\": null,\n"
                "  \"tissues_examined_and_stained\": [\n"
                "    {\n"
                "      \"site\": \"<site_name>\",\n"
                "      \"details\": [\n"
                "        \"<details_about_tissue>\"\n"
                "      ]\n"
                "    }\n"
                "  ],\n"
                "  \"tumor_stage_or_grade\": null,\n"
                "  \"general_cancer_type\": null,\n"
                "  \"diagnosis_or_type_of_cancer\": null,\n"
                "  \"type_of_procedure_performed\": [\n"
                "    \"<procedure_description>\"\n"
                "  ],\n"
                "  \"metastases\": null,\n"
                "  \"genetic_metadata\": {\n"
                "    \"tumor_type\": [],\n"
                "    // ER: Include only if ER data is found\n"
                "    // PR: Include only if PR data is found\n"
                "    // HER2: Include only if HER2 data is found\n"
                "    // Ki67: Include only if Ki67 data is found\n"
                "    // Any other relevant data\n"
                "    // recurrence_score: Include only if recurrence score data is found\n"
                "  },\n"
                "  \"comments\": \"pathology text too long to extract. Please extract manually\"  // Only if phrase is present in the text\n"
                "}\n"
                "```"
            )
        },
        {
            "role": "user",
            "content": report_text
        }
    ]

    try:
        # Make the API call
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Adjust based on your model version
            messages=messages,
            max_tokens=1000  # Adjust based on expected output length
        )

        # Extract the generated response text and return it
        return completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None


def analyze_all_texts_report(report_text):
    """
    Send the report text to OpenAI API for analysis to specifically extract relevant information.
    """

    # Instantiate the OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Constructing the message format for the OpenAI chat API
    messages = [
        {
            "role": "system",
            "content": (
                "You are analyzing and extracting information from a patient's medical records in Hebrew. Extract the "
                "following information and output the results in English:\n"
                "1. Ethnicity (if Jewish, specify whether Sephardi, Ashkenazi, Mizrahi, etc., or the country from "
                "which they or their family migrated to Israel).\n"
                "2. Any family history related to diseases, pathologies, allergies, etc., categorized by side "
                "(paternal/maternal), the relative (e.g., aunt, uncle, etc.), the specific condition with age of onset "
                "if provided, and any genetic mutations mentioned.\n"
                "3. Allergies (drug-related and non-drug-related), each as a list of strings.\n"
                "4. Tumor stage or grade information, which includes details about tumor characteristics, histological "
                "grade, and other descriptive features related to tumor staging or grade.\n"
                "5. Any **pTNM results** (e.g., pT3N1M0) as a distinct entry. Ensure these pTNM values are extracted "
                "separately from tumor stage or grade details if they appear in the report.\n"
                "6. The general category of cancer type, matching one of these: Breast Cancer, Pancreatic Cancer, "
                "Colorectal Cancer, Lung Cancer, Ovarian Cancer, Endometrial Cancer. If no match fits, provide the response as the LLM determines.\n"
                "7. Any genetic data mentioned (specific genes expressed, mutations, etc., e.g., Ki-67, ER, PR, HER-2, "
                "BRCA, etc.).\n"
                "8. If metastases were found and mentioned.\n"
                "9. Any other relevant genetic data or metadata found in the medical texts.\n\n"
                "For the human-readable summary, provide the extracted information in the same order as the list above, "
                "with each item labeled using the following format:\n"
                "1. **Ethnicity**: <extracted_data>.\n"
                "2. **Family History**: <extracted_data>.\n"
                "...\n"
                "If any information is missing in the report, write 'Not provided in the report' instead.\n\n"
                "Please provide the output in two clearly separated sections:\n"
                "1. A human-readable summary (labeled 'Human-readable Summary').\n"
                "2. A **complete and valid JSON** response (labeled 'JSON Format'), with no truncation or incomplete parts.\n\n"
                "Ensure the JSON structure includes the following specific format for genetic_metadata, only including fields that are found in the report:\n"
                "\n"
                "```json\n"
                "{\n"
                "  \"ethnicity\": null,\n"
                "  \"family_history\": {\n"
                "    \"paternal\": [\n"
                "      {\n"
                "        \"relative\": null,\n"
                "        \"condition\": null,\n"
                "        \"age_of_onset\": null,\n"
                "        \"mutations\": []\n"
                "      }\n"
                "    ],\n"
                "    \"maternal\": [\n"
                "      {\n"
                "        \"relative\": null,\n"
                "        \"condition\": null,\n"
                "        \"age_of_onset\": null,\n"
                "        \"mutations\": []\n"
                "      }\n"
                "    ]\n"
                "  },\n"
                "  \"allergies\": {\n"
                "    \"drug_related\": [],\n"
                "    \"non_drug_related\": []\n"
                "  },\n"
                "  \"tumor_stage_or_grade\": null,\n"
                "  \"ptnm_results\": null,\n"
                "  \"general_cancer_type\": null,\n"
                "  \"genetic_metadata\": {\n"
                "    \"tumor_type\": [],  // Only include if tumor type data is found\n"
                "    // ER: Include only if ER data is found\n"
                "    // PR: Include only if PR data is found\n"
                "    // HER2: Include only if HER2 data is found\n"
                "    // Ki67: Include only if Ki67 data is found\n"
                "    // Any other relevant data\n"
                "    // recurrence_score: Include only if recurrence score data is found\n"
                "  },\n"
                "  \"metastases\": null\n"
                "}\n"
                "```"
            )
        },
        {
            "role": "user",
            "content": report_text
        }
    ]

    try:
        # Make the API call
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Adjust based on your model version
            messages=messages,
            max_tokens=1000  # Adjust based on expected output length
        )

        # Extract the generated response text and return it
        return completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None


def combine_batched_analyses(batched_analyses):
    """
    Combines multiple batched analyses into a single, coherent report.

    Args:
        batched_analyses (list of str): Each string in the list represents a batched analysis.

    Returns:
        str: A combined analysis preserving the structure of each batch type.
    """

    # Instantiate the OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Join batched analyses with clear separation for GPT to understand and combine effectively
    combined_analysis_input = "\n\n--- Next Batch ---\n\n".join(batched_analyses)

    # Messages that guide GPT to combine the batched responses into a single summary
    messages = [
        {
            "role": "system",
            "content": (
                "You are a medical data summarization assistant. You will receive multiple analysis reports generated"
                " in batches for a single patient, each covering part of their medical history or their pathology"
                " report. Your task is to combine all of the following reports into a single, coherent analysis.\n\n"
                "IMPORTANT:\n"
                "1. The structure and format of each batch’s JSON must be preserved exactly, with no changes to key"
                " names, nested levels, or data types. Each JSON key should appear as given, even if some keys contain"
                " 'null' values. This applies to the format of the human readable summary as well.\n"
                "2. Only the contents within each JSON key can be consolidated. If duplicate information appears,"
                " merge it appropriately, but do not remove any JSON keys or alter their order.\n\n"
                "STRUCTURE:\n"
                "- Preserve the original formatting or layout style of each batch, simply merging relevant information"
                " from multiple batches into one report.\n"
                "- Retain the order of data as it appears within each batch type, and ensure any repetitive"
                " information is summarized without modifying the JSON structure.\n\n"
                "Summarize the human-readable text while keeping the JSON rigid and consistent with the original"
                " layout. Each JSON output should strictly follow the original structure without any modifications to"
                " key names, order, or nesting."
            )
        },
        {
            "role": "user",
            "content": combined_analysis_input
        }
    ]

    try:
        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Adjust based on your model version
            messages=messages,
            max_tokens=1000  # Adjust based on expected output length
        )

        # Extract the response
        combined_analysis = response.choices[0].message.content.strip()
        return combined_analysis

    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None
