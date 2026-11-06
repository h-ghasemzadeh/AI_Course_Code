import pandas as pd
from docx import Document

# Read the CSV file
df = pd.read_csv("citations.csv")

# Create a Word document
doc = Document()

# Add a table to the document
table = doc.add_table(rows=df.shape[0] + 1, cols=df.shape[1])

# Add the header row
for col_idx, column in enumerate(df.columns):
    table.cell(0, col_idx).text = column

# Add the data rows
for row_idx, row in df.iterrows():
    for col_idx, value in enumerate(row):
        table.cell(row_idx + 1, col_idx).text = str(value)

# Save the document
doc.save("citations.docx")