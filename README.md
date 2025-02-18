OVERVIEW:


The Disease Diagnosis Bot is an AI-powered medical assistant designed to help healthcare professionals and individuals diagnose diseases based on symptoms. It leverages advanced natural language processing (NLP) and machine learning models to provide accurate and concise disease predictions along with relevant medical information.

The bot processes user-inputted symptoms, matches them against a vast 15,000+ page medical dataset, and retrieves relevant disease details using BioBERT and Sentence-BERT for medical text understanding. It also uses Hugging Face transformers for generating summarized insights, ensuring that users receive reliable and concise information.



DATA 
Book: Harrison's Principle of internal medicine 21st edition Vol.1 (this book is used as data.pdf in the code)

Link: https://notesmed.com/harrison-principles-of-internal-medicine-21st-edition/




TECHNOLOGIES USED:

Machine Learning & NLP: BioBERT, Sentence-BERT, Hugging Face Transformers

Search & Retrieval: Whoosh (lightweight text search engine)

Backend & Deployment: Python, Flask/FastAPI

Data Processing: SpaCy, NLTK  



HOW IT WORKS:

User Inputs Symptoms – The bot collects symptoms entered by the user.

Symptom Matching – Uses BioBERT and Sentence-BERT to identify the closest matching diseases.

Disease Prediction & Ranking – Retrieves potential diseases and ranks them based on similarity scores.

Summarization – Provides a concise summary of the disease using Hugging Face models.

Response Generation – Displays the results with disease name, symptoms, causes, and possible treatments.
