import os
import json
import numpy as np
import torch
from transformers import pipeline, BertTokenizer, BertForTokenClassification
from sentence_transformers import SentenceTransformer, util
import spacy
from whoosh.index import open_dir
from whoosh.qparser import QueryParser

# Load spaCy model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Load the BioBERT model and tokenizer for NER (Named Entity Recognition)
bio_model_name = "dmis-lab/biobert-v1.1"
bio_tokenizer = BertTokenizer.from_pretrained(bio_model_name)
bio_model = BertForTokenClassification.from_pretrained(bio_model_name)

# Load the summarization model using Hugging Face pipeline
summarizer = pipeline("summarization")

# Load Sentence-BERT model for calculating semantic similarity
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Whoosh index directory path
index_dir = "index"

#Directory where disease summary JSON files are stored
output_dir = "disease_summaries"


def load_disease_summaries():
    """Load all disease summaries from the JSON files in the output directory."""
    disease_data = {}
    for filename in os.listdir(output_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                disease_name = list(data.keys())[0]  # Assume the disease name is the first key in JSON
                disease_data[disease_name] = data[disease_name]
    return disease_data


def query_symptoms(symptoms_query, disease_data):
    """Query diseases based on symptoms using semantic similarity."""
    symptoms_query = symptoms_query.lower()

    # Prepare to store results and embeddings
    matching_diseases = []
    disease_embeddings = []
    disease_names = []

    for disease, info in disease_data.items():
        symptoms = info.get('symptoms', '')
        if not symptoms:
            continue  # Skip if no symptoms are available
        
        # Generate embedding for disease symptoms
        embedding = sentence_model.encode(symptoms, convert_to_tensor=True)
        disease_embeddings.append(embedding)
        disease_names.append(disease)

    if not disease_embeddings:
        return []

    # Stack embeddings into a single tensor
    disease_embeddings_tensor = torch.stack(disease_embeddings)

    # Generate embedding for the query symptoms
    query_embedding = sentence_model.encode(symptoms_query, convert_to_tensor=True)

    # Calculate cosine similarity between query and disease embeddings
    similarities = util.pytorch_cos_sim(query_embedding, disease_embeddings_tensor)
    best_matches = similarities[0].cpu().numpy()  # Convert to numpy array

    # Set a similarity threshold for filtering matching diseases
    matching_threshold = 0.5
    matching_indices = np.where(best_matches > matching_threshold)[0]

    for idx in matching_indices:
        matching_diseases.append({
            'disease': disease_names[idx],
            'symptoms': disease_data[disease_names[idx]].get('symptoms', 'N/A'),
            'cause': disease_data[disease_names[idx]].get('cause', 'Unknown'),
            #'class': disease_data[disease_names[idx]].get('class', 'Not Specified'),
            'similarity_score': best_matches[idx]
        })

    # Sort the matching diseases by similarity score in descending order
    matching_diseases = sorted(matching_diseases, key=lambda x: x['similarity_score'], reverse=True)

    return matching_diseases


def query_whoosh_index(query_text):
    """Search for matching documents in the Whoosh index."""
    ix = open_dir(index_dir)
    parser = QueryParser("content", ix.schema)
    
    with ix.searcher() as searcher:
        query = parser.parse(query_text)
        results = searcher.search(query, limit=5)  # Retrieve top 5 relevant documents
        return [hit['path'] for hit in results]


def get_info_from_whoosh(query):
    """Retrieve relevant file paths from the Whoosh index."""
    file_paths = query_whoosh_index(query)
    return file_paths



if __name__ == "__main__":
    # Load the disease summaries into memory
    disease_data = load_disease_summaries()
    
    # Example: Prompt user to input symptoms
    symptoms_query = input("Enter symptoms to query diseases: ")
    
    # Query disease summaries for matching diseases based on symptoms
    results = query_symptoms(symptoms_query, disease_data)
    
    if results:
        for result in results:
            print(f"Disease: {result['disease']}")
            print(f"Symptoms: {result['symptoms']}")
            print(f"Cause: {result['cause']}")
            
            print(f"Similarity Score: {result['similarity_score']:.4f}")
            print("-" * 40)
    else:
        print("No matching diseases found in summaries. Searching in Whoosh index...")
        whoosh_results = get_info_from_whoosh(symptoms_query)
        
        if whoosh_results:
            print("Found related documents in Whoosh index:")
            for path in whoosh_results:
                print(f"Document: {path}")
        else:
            print("No related documents found in Whoosh index.")

