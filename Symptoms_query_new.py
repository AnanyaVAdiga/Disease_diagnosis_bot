import json
import torch
from transformers import BertTokenizer, BertForMaskedLM
from sentence_transformers import SentenceTransformer, util

# Load the BioBERT model and tokenizer for similarity calculation
bio_model_name = "dmis-lab/biobert-v1.1"
bio_tokenizer = BertTokenizer.from_pretrained(bio_model_name)
bio_model = BertForMaskedLM.from_pretrained(bio_model_name)

# Use a general sentence transformer model for embedding comparison
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Path to the JSON file containing disease and symptoms
disease_summary_file = "new.json"

def load_disease_summaries():
    """Load all disease summaries from the JSON file."""
    with open(disease_summary_file, 'r', encoding='utf-8') as f:
        disease_data = json.load(f)  # Load JSON file as a dictionary
    return disease_data


def bio_bert_embed(text):
    """Generate embeddings using BioBERT model."""
    inputs = bio_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        # Set output_hidden_states=True to get the hidden states
        outputs = bio_model(**inputs, output_hidden_states=True)
        
    # Use the hidden states (the last layer's output)
    hidden_states = outputs.hidden_states[-1]  # The last layer's hidden states
    embeddings = hidden_states.mean(dim=1)  # Mean pooling over tokens to get the sentence-level embedding
    return embeddings

def query_symptoms(symptoms_query, disease_data):
    """Query diseases based on symptoms provided by the user using BioBERT."""
    # Convert the query symptoms into a list
    symptoms_query = symptoms_query.lower().split(",")  # Split query into list of symptoms
    
    # Generate embeddings for each symptom in the query using BioBERT
    query_embeddings = [bio_bert_embed([symptom]) for symptom in symptoms_query]
    
    # Average the embeddings for the query (mean pooling across all symptoms)
    query_embedding = torch.mean(torch.stack(query_embeddings), dim=0)

    matching_diseases = []

    for disease, details in disease_data.items():
        disease_symptoms = [symptom.lower() for symptom in details["Symptoms"]]  # Normalize symptoms to lowercase

        # Calculate the similarity score based on the common symptoms
        matching_symptoms = set(symptoms_query).intersection(disease_symptoms)
        if matching_symptoms:  # If there's at least one matching symptom
            # Generate embeddings for the matching disease symptoms
            disease_embeddings = [bio_bert_embed([symptom]) for symptom in matching_symptoms]
            
            # Average the embeddings for the disease (mean pooling across matched symptoms)
            disease_embedding = torch.mean(torch.stack(disease_embeddings), dim=0)

            # Calculate cosine similarity between the query symptoms and disease symptoms
            similarity_score = util.pytorch_cos_sim(query_embedding, disease_embedding).item()

            # If similarity score is greater than a threshold, consider it a match
            if similarity_score > 0.5:  # You can adjust this threshold
                matching_diseases.append({
                    'disease': disease,
                    'symptoms': details["Symptoms"],
                    'matching_symptoms': list(matching_symptoms),  # List of symptoms that match
                    'similarity_score': similarity_score
                })

    return matching_diseases


if __name__ == "__main__":
    # Load the disease summaries into memory
    try:
        disease_data = load_disease_summaries()
    except FileNotFoundError:
        print(f"Error: File '{disease_summary_file}' not found. Please check the path.")
        exit(1)

    # Example: Prompt user to input symptoms
    symptoms_query = input("Enter symptoms (comma separated) to query diseases: ")

    # Query disease summaries for matching diseases based on symptoms
    results = query_symptoms(symptoms_query, disease_data)

    if results:
        print(f"Found the following diseases related to the symptoms: {symptoms_query}")
        for result in results:
            print(f"\nDisease: {result['disease']}")
            print(f"Symptoms: {', '.join(result['symptoms'])}")
            print(f"Matching Symptoms: {', '.join(result['matching_symptoms'])}")
            print(f"Similarity Score: {result['similarity_score']:.4f}")
            print("-" * 40)
    else:
        print("No diseases found related to the entered symptoms.")





# import json
# import torch
# from transformers import BertTokenizer, BertForMaskedLM
# from sentence_transformers import SentenceTransformer, util

# # Load the BioBERT model and tokenizer for similarity calculation
# bio_model_name = "dmis-lab/biobert-v1.1"
# bio_tokenizer = BertTokenizer.from_pretrained(bio_model_name)
# bio_model = BertForMaskedLM.from_pretrained(bio_model_name)

# # Use a general sentence transformer model for embedding comparison
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Path to the JSON file containing disease and symptoms
# disease_summary_file = "new.json"

# def load_disease_summaries():
#     """Load all disease summaries from the JSON file."""
#     with open(disease_summary_file, 'r', encoding='utf-8') as f:
#         disease_data = json.load(f)  # Load JSON file as a dictionary
#     return disease_data


# def bio_bert_embed(text):
#     """Generate embeddings using BioBERT model."""
#     inputs = bio_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         # Set output_hidden_states=True to get the hidden states
#         outputs = bio_model(**inputs, output_hidden_states=True)
        
#     # Use the hidden states (the last layer's output)
#     hidden_states = outputs.hidden_states[-1]  # The last layer's hidden states
#     embeddings = hidden_states.mean(dim=1)  # Mean pooling over tokens to get the sentence-level embedding
#     return embeddings

# def query_symptoms(symptoms_query, disease_data):
#     """Query diseases based on symptoms provided by the user using a weighted similarity score."""
#     symptoms_query = symptoms_query.lower().split(",")  # Split query into list of symptoms
#     symptoms_query = [symptom.strip() for symptom in symptoms_query]  # Remove extra spaces

#     matching_diseases = []

#     for disease, details in disease_data.items():
#         disease_symptoms = [symptom.lower() for symptom in details["Symptoms"]]  # Normalize symptoms to lowercase

#         # Find the matching symptoms
#         matched_symptoms = [symptom for symptom in symptoms_query if symptom in disease_symptoms]

#         # Calculate similarity scores
#         if matched_symptoms:
#             query_match_ratio = len(matched_symptoms) / len(symptoms_query)
#             disease_match_ratio = len(matched_symptoms) / len(disease_symptoms)
#             weighted_similarity_score = query_match_ratio * disease_match_ratio

#             matching_diseases.append({
#                 'disease': disease,
#                 'symptoms': details["Symptoms"],
#                 'matching_symptoms': matched_symptoms,
#                 'similarity_score': weighted_similarity_score
#             })

#     return sorted(matching_diseases, key=lambda x: x['similarity_score'], reverse=True)


# if __name__ == "__main__":
#     # Load the disease summaries into memory
#     try:
#         disease_data = load_disease_summaries()
#     except FileNotFoundError:
#         print(f"Error: File '{disease_summary_file}' not found. Please check the path.")
#         exit(1)

#     # Example: Prompt user to input symptoms
#     symptoms_query = input("Enter symptoms (comma separated) to query diseases: ")

#     # Query disease summaries for matching diseases based on symptoms
#     results = query_symptoms(symptoms_query, disease_data)

#     if results:
#         print(f"Found the following diseases related to the symptoms: {symptoms_query}")
#         for result in results:
#             print(f"\nDisease: {result['disease']}")
#             print(f"Symptoms: {', '.join(result['symptoms'])}")
#             print(f"Matching Symptoms: {', '.join(result['matching_symptoms'])}")
#             print(f"Similarity Score: {result['similarity_score']:.4f}")
#             print("-" * 40)
#     else:
#         print("No diseases found related to the entered symptoms.")
