import pandas as pd

from metadata_scripts.helpers import get_file_path

# Load the Excel file
file_path = get_file_path('sample Metadata (2)')
df = pd.read_excel(file_path)

# Convert column names to uppercase
df.columns = df.columns.str.upper()

# Print the column names
print(df.columns)
