import pdfplumber

from metadata_scripts.helpers import get_file_path

file_path = get_file_path("pdf_test", True)

# Open the PDF file
with pdfplumber.open(file_path) as pdf:
    # Loop through each page in the PDF
    for i, page in enumerate(pdf.pages):
        print(f"Page {i + 1}")
        # Extract tables from the page
        tables = page.extract_table()
        if tables:
            # Print the table (as a list of rows)
            for row in tables:
                print(row)
