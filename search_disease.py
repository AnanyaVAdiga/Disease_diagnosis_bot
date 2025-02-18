# search_disease.py
import os
from whoosh.index import open_dir
from whoosh.qparser import QueryParser

# Paths
index_dir = "C:\\Users\\ananya\\Desktop\\doctor\\ouput\\whoosh_index2"

# Function to search for a disease using Whoosh
def search_disease_with_whoosh(disease_name, index_directory):
    ix = open_dir(index_directory)
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse(disease_name)
        results = searcher.search(query)
        return [(result['file_path'], result['category']) for result in results]

# Example usage
if __name__ == "__main__":
    disease_to_search = input("Enter the disease to search for: ")
    result_files = search_disease_with_whoosh(disease_to_search, index_dir)

    if result_files:
        print(f"Disease '{disease_to_search}' found in the following files:")
        for file_path, category in result_files:
            print(f"- {file_path} (Category: {category})")
    else:
        print(f"Disease '{disease_to_search}' not found in any file.")
