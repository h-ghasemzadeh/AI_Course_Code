import pandas as pd
from docx import Document

# Read the CSV file
df = pd.read_csv("citations.csv")

# Create a Word document
doc = Document()

# Function to format a single row in APA style
def format_apa(row):
    # Example for journal articles
    citation = f" {row['Year']}. {row['Title']}, {row['Publication']}, {row['Volume']}, {row['Number']}, {row['Pages']}, {row['Publisher']}"
    return citation

# Add each citation to the Word document
for _, row in df.iterrows():
    apa_citation = format_apa(row)
    doc.add_paragraph(apa_citation)
    doc.add_paragraph()  # Add a blank line between citations



# Save the document
doc.save("c:/lecture/AI/support/code/citations.docx")
