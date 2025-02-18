from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in
import os

# Define a schema including the 'path' field
schema = Schema(path=ID(stored=True), content=TEXT(stored=True))

# Directory where the index will be stored
index_dir = "C:\\Users\\ananya\\Desktop\\doctor\\ouput\\index4"

# Create the index directory if it doesn't exist
if not os.path.exists(index_dir):
    os.mkdir(index_dir)

# Create the index
ix = create_in(index_dir, schema)

# Sample indexing of files
def add_files_to_index(ix, folder_path):
    writer = ix.writer()
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                writer.add_document(path=file_path, content=content)
    
    writer.commit()

# Index the folder containing the files 
add_files_to_index(ix, "C:\\Users\\ananya\\Desktop\\doctor\\ouput\\cleaned_data.txt")
