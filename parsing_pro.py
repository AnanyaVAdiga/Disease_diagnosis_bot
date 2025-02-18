# import os
# import json
# import torch
# from transformers import pipeline,  BertTokenizer, BertForTokenClassification
# from whoosh.index import open_dir
# from whoosh.qparser import QueryParser
# import spacy
# from concurrent.futures import ProcessPoolExecutor, as_completed

# # Load spaCy model
# nlp = spacy.load("en_core_web_sm")

# # Load the distilled model and tokenizer for NER
# bio_model_name = "dmis-lab/biobert-v1.1"
# bio_tokenizer = BertTokenizer.from_pretrained(bio_model_name)
# bio_model = BertForTokenClassification.from_pretrained(bio_model_name)

# # Load summarization model
# summarizer = pipeline("summarization")

# # Whoosh index directory path
# index_dir = "C:\\Users\\ananya\\Desktop\\doctor\\ouput\\index"

# # Output directory for individual disease files
# output_dir = "disease_summaries3"

# def query_whoosh_index(query_text):
#     ix = open_dir(index_dir)
#     parser = QueryParser("content", ix.schema)
    
#     with ix.searcher() as searcher:
#         query = parser.parse(query_text)
#         results = searcher.search(query, limit=None)  # Get all relevant results
#         return [hit['path'] for hit in results]

# def extract_disease_info(text):
#     doc = nlp(text)

#     disease_info = {
#         'symptoms': [],
#         'cause': [],
#         'class': []
#     }

#     keywords = {
#         'symptoms' : ["symptom", "sign", "indication", "symptomatic", "effect", "condition", "problem", "difficulty"],
#         'cause': ["cause", "etiology", "origin", "source", "trigger", "contributor", "risk factor", "pathogen", "agent", "factor", "reason"],
#         'class' : ["class", "type", "category", "classification", "viral", "bacterial", "fungal", "allergic", "autoimmune", "genetic", "chronic", "acute"]
#     }

#     for sent in doc.sents:
#         sent_lower = sent.text.lower()
#         for category, word_list in keywords.items():
#             if any(keyword in sent_lower for keyword in word_list):
#                 disease_info[category].append(sent.text.strip())

#     return disease_info

# def summarize_texts(texts):
#     if not texts:
#         return ""
    
#     combined_text = " ".join(texts)
#     if len(combined_text) > 1024:
#         combined_text = combined_text[:1024]

#     summarized = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)
#     return summarized[0]['summary_text']

# def process_single_disease(disease):
#     file_paths = query_whoosh_index(disease)
#     if not file_paths:
#         print(f"No files found for disease: {disease}")
#         return None

#     all_symptoms = []
#     all_causes = []
#     all_classes = []

#     for file_path in file_paths:
#         try:
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 text = file.read()
#             disease_info = extract_disease_info(text)
#             all_symptoms.extend(disease_info['symptoms'])
#             all_causes.extend(disease_info['cause'])
#             all_classes.extend(disease_info['class'])
#         except Exception as e:
#             print(f"Error processing {file_path}: {e}")

#     summarized_results = {
#         'symptoms': summarize_texts(all_symptoms),
#         'cause': summarize_texts(all_causes),
#         'class': summarize_texts(all_classes)
#     }

#     return summarized_results

# def save_disease_info(disease, info):
#     filename = "".join(x for x in disease if x.isalnum() or x in [' ', '-', '_']).rstrip()
#     filename = filename.replace(' ', '_') + '.json'
    
#     filepath = os.path.join(output_dir, filename)
    
#     with open(filepath, 'w', encoding='utf-8') as f:
#         json.dump({disease: info}, f, indent=4)

# def process_diseases_list(diseases_list):
#     os.makedirs(output_dir, exist_ok=True)

#     with ProcessPoolExecutor() as executor:
#         futures = {executor.submit(process_single_disease, disease): disease for disease in diseases_list}
        
#         for future in as_completed(futures):
#             disease = futures[future]
#             try:
#                 summarized_info = future.result()
#                 if summarized_info:
#                     save_disease_info(disease, summarized_info)
#                 print(f"Completed processing: {disease}")
#             except Exception as e:
#                 print(f"Error processing disease {disease}: {e}")

# if __name__ == "__main__":
#     with open("disease_list.json", 'r') as f:
#         data = json.load(f)
#         diseases_list = data["Diseases"]

#     process_diseases_list(diseases_list)
#     print(f"Summarized results for all diseases saved in individual files in the '{output_dir}' directory")                 





import os
import json
import torch
from transformers import pipeline, BertTokenizer, BertForTokenClassification
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
import spacy
from concurrent.futures import ProcessPoolExecutor, as_completed

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the distilled model and tokenizer for NER
bio_model_name = "dmis-lab/biobert-v1.1"
bio_tokenizer = BertTokenizer.from_pretrained(bio_model_name)
bio_model = BertForTokenClassification.from_pretrained(bio_model_name)

# Load summarization model
summarizer = pipeline("summarization")

# Whoosh index directory path
index_dir = "index"

# Output directory for individual disease files
output_dir = "disease_summaries3"

def query_whoosh_index(query_text):
    ix = open_dir(index_dir)
    parser = QueryParser("content", ix.schema)
    
    with ix.searcher() as searcher:
        query = parser.parse(query_text)
        results = searcher.search(query, limit=None)  # Get all relevant results
        return [hit['path'] for hit in results]

def extract_disease_info(text):
    doc = nlp(text)

    disease_info = {
        'symptoms': [],
        'cause': [],
        'class': []
    }

    keywords = {
        'symptoms': ["symptom", "sign", "indication", "symptomatic", "effect", "condition", "problem", "difficulty"],
        'cause': ["cause", "etiology", "origin", "source", "trigger", "contributor", "risk factor", "pathogen", "agent", "factor", "reason"],
        'class': ["class", "type", "category", "classification", "viral", "bacterial", "fungal", "allergic", "autoimmune", "genetic", "chronic", "acute"]
    }

    for sent in doc.sents:
        sent_lower = sent.text.lower()
        for category, word_list in keywords.items():
            if any(keyword in sent_lower for keyword in word_list):
                disease_info[category].append(sent.text.strip())

    return disease_info

def summarize_texts(texts):
    if not texts:
        return ""
    
    combined_text = " ".join(texts)
    if len(combined_text) > 1024:
        combined_text = combined_text[:1024]

    summarized = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)
    return summarized[0]['summary_text']

def process_single_disease(disease):
    file_paths = query_whoosh_index(disease)
    if not file_paths:
        print(f"No files found for disease: {disease}")
        return None

    all_symptoms = []
    all_causes = []
    all_classes = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            disease_info = extract_disease_info(text)
            all_symptoms.extend(disease_info['symptoms'])
            all_causes.extend(disease_info['cause'])
            all_classes.extend(disease_info['class'])
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    summarized_results = {
        'symptoms': summarize_texts(all_symptoms),
        'cause': summarize_texts(all_causes),
        'class': summarize_texts(all_classes)
    }

    return summarized_results

def save_disease_info(disease, info):
    filename = "".join(x for x in disease if x.isalnum() or x in [' ', '-', '_']).rstrip()
    filename = filename.replace(' ', '_') + '.json'
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({disease: info}, f, indent=4)

def process_diseases_list(diseases_list):
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 100
    for i in range(0, len(diseases_list), batch_size):
        batch = diseases_list[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} with {len(batch)} diseases")
        
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_single_disease, disease): disease for disease in batch}
            
            for future in as_completed(futures):
                disease = futures[future]
                try:
                    summarized_info = future.result()
                    if summarized_info:
                        save_disease_info(disease, summarized_info)
                    print(f"Completed processing: {disease}")
                except Exception as e:
                    print(f"Error processing disease {disease}: {e}")

if __name__ == "__main__":
    with open("CDisease_List.json", 'r') as f:
        data = json.load(f)
        diseases_list = data["disease"]

    process_diseases_list(diseases_list)
    print(f"Summarized results for all diseases saved in individual files in the '{output_dir}' directory")


