from translate import Translator

text = "עישון: לא מעשן/ת, שינה: לא נוטל/ת תרופות לשינה, סמים: לא נוטל/ת סמים, אלכוהול: לא שותה אלכוהול, פעילות גופנית: כן (פרט)"


def translate_sentence(text, from_lang="hebrew", to_lang="english"):
    """
    Translates a sentence from one language to another using the `translate` package.

    Args:
        text (str): The sentence to translate.
        from_lang (str): The source language (default is "hebrew").
        to_lang (str): The target language (default is "english").

    Returns:
        str: The translated sentence.
    """
    translator = Translator(from_lang=from_lang, to_lang=to_lang)

    # Split the text into sections using commas to handle each part individually
    sections = text.split(', ')

    # Translate each section
    translated_sections = [translator.translate(section) for section in sections]

    # Join the translated sections back into a single string
    translated_text = ', '.join(translated_sections)

    return translated_text
