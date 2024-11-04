import os
import re

import pandas as pd
import tiktoken


def clean_keys(data_dict):
    """Recursively clean up keys in the dictionary."""
    cleaned_dict = {}
    for key, value in data_dict.items():
        # Clean up the key
        clean_key = key.replace(':_x000D_', '').replace('\n', '').strip()

        # If the value is a nested dictionary, apply the same cleaning recursively
        if isinstance(value, dict):
            cleaned_dict[clean_key] = clean_keys(value)
        else:
            cleaned_dict[clean_key] = value

    return cleaned_dict


def get_file_path(file_name: str):
    # Get the current script directory
    script_dir = os.path.dirname(__file__)

    # Construct the path to the file in the 'data' folder
    file_path = os.path.join(script_dir, '..', 'data', f'{file_name}.xlsx')

    # Normalize the path (optional)
    return os.path.abspath(file_path)


def fix_invalid_dates(date_str):
    """
    Function to manually fix invalid dates, if possible.
    E.g., if a date has missing parts, we can infer or fix them here.
    """
    # If the date has extra spaces, trim them
    if pd.isna(date_str):
        return None

    date_str = date_str.strip()

    # Example: Fix incomplete year (e.g., "31.10." -> "31.10.22" if the year is missing)
    if date_str.count('.') == 2 and len(date_str.split('.')[-1]) < 2:
        date_str += '22'  # Assuming it's the current century, adjust as needed

    # Add more custom rules for fixing dates if necessary
    return date_str


def clean_text_with_actual_newlines(text):
    """
    Clean text fields by replacing placeholders with actual new lines ('\n') and removing redundant new lines.
    """
    if isinstance(text, list):
        return [re.sub(r'(\n)+', '\n', t.replace('_x000D_', '\n').replace('<NEW_LINE>', '\n')).rstrip('\n').strip() for
                t in text]
    elif isinstance(text, str):
        return re.sub(r'(\n)+', '\n', text.replace('_x000D_', '\n').replace('<NEW_LINE>', '\n')).rstrip('\n').strip()
    return text


def count_tokens(text):
    """
    Count the number of tokens in the given text using the GPT-like tokenizer.
    """
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    return len(encoding.encode(text))


def split_text_into_batches(text, max_tokens=120000):
    """Split text into batches that don't exceed the maximum token limit."""
    words = text.split()
    batches = []
    current_batch = []
    current_batch_tokens = 0

    for word in words:
        word_tokens = count_tokens(word)

        # Start a new batch if adding this word exceeds the max token limit
        if current_batch_tokens + word_tokens > max_tokens:
            batches.append(' '.join(current_batch))
            current_batch = []
            current_batch_tokens = 0

        # Add word to the current batch
        current_batch.append(word)
        current_batch_tokens += word_tokens

    # Add the last batch
    if current_batch:
        batches.append(' '.join(current_batch))

    return batches


def format_data(data):
    """
    Formats data based on its type (dictionary, list, or string),
    converting dictionary keys to a more readable format by replacing
    underscores with spaces and capitalizing words. Handles nested
    dictionaries and lists by recursively formatting them, and replaces
    empty lists with 'None'.

    Args:
        data (dict, list, or str): The input data to format.

    Returns:
        str: A formatted string representation of the input data.
    """
    formatted_string = ""

    if isinstance(data, dict):
        # Iterate through dictionary and format as Key: Value
        for key, value in data.items():
            # Replace underscores with spaces and capitalize each word
            formatted_key = key.replace('_', ' ').title()

            # Check if the value is a nested dictionary
            if isinstance(value, dict):
                # Recursively format the nested dictionary
                nested_data = format_data(value)
                formatted_string += f"{formatted_key}:\n  {nested_data}\n"  # Indent nested values

            # If value is a list, handle each item in the list
            elif isinstance(value, list):
                # Join list elements with commas if not empty, otherwise use 'None'
                if value:
                    formatted_list = ""
                    for item in value:
                        if isinstance(item, dict):
                            # Recursively format dictionary items in the list
                            formatted_list += f"\n  {format_data(item)}\n"
                        else:
                            formatted_list += f"{item}, "
                    formatted_string += f"{formatted_key}:{formatted_list.rstrip(', ')}\n"
                else:
                    formatted_string += f"{formatted_key}: None\n"

            else:
                # Format the value directly
                formatted_string += f"{formatted_key}: {value}\n"

    elif isinstance(data, list):
        # Join list elements with commas or replace with 'None' if list is empty
        formatted_string = ", ".join(map(str, data)) if data else "None"

    elif isinstance(data, str):
        # Return the string as is
        formatted_string = data
    else:
        # If it's not a dict, list, or string, return it as a string
        formatted_string = str(data)

    return formatted_string.strip()  # Strip any trailing newlines or spaces
