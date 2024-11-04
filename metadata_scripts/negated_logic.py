import re


def clean_text_with_placeholders(text):
    """
    Clean text fields by replacing line breaks with '<NEW_LINE>' for better readability and reduced token load.
    """
    if isinstance(text, list):
        return [re.sub(r'(<NEW_LINE>)+', '<NEW_LINE>',
                       t.replace('_x000D_', '<NEW_LINE>').replace('\n', '<NEW_LINE>')).rstrip('<NEW_LINE>').strip() for
                t in text]
    elif isinstance(text, str):
        return re.sub(r'(<NEW_LINE>)+', '<NEW_LINE>',
                      text.replace('_x000D_', '<NEW_LINE>').replace('\n', '<NEW_LINE>')).rstrip('<NEW_LINE>').strip()
    return text
