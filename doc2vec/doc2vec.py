# doc2vec.py

# this script allows the user to upload a document and it will output the top 3 TCLP clause matches

# necessary imports
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk

# Loading in the clauses
# Note: this requires that you have the England and Wales clauses in a folder called 'England:Wales' in the same directory as this script
folder_path = "England:Wales"

documents = []
file_names = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and filename.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            documents.append(file.read())
            file_names.append(filename)

# tokenizing the clauses
tagged_clauses = [
    TaggedDocument(words=word_tokenize(clause.lower()), tags=[str(i)])
    for i, clause in enumerate(documents)
]

# training the model
# NOTE TO SELF: This could be done outside of this file, instead loading in the model each time
model = Doc2Vec(vector_size=50, min_count=1, epochs=20)
model.build_vocab(tagged_clauses)
model.train(tagged_clauses, total_examples=model.corpus_count, epochs=model.epochs)

# computing the document vectors
document_vectors = [model.dv[str(i)] for i in range(len(documents))]


# function to get the top 3 matches
def find_best_matching_document(
    query, model, document_vectors, documents, verbose=False, top_three=False
):
    # Infer the vector for the query document
    query_vector = model.infer_vector(word_tokenize(query.lower()))

    # Compute cosine similarity with precomputed document vectors
    query_vector = query_vector.reshape(1, -1)
    similarities = cosine_similarity(query_vector, document_vectors).flatten()

    # Find the best match
    best_match_index = np.argmax(similarities)

    # find the top three matches
    if top_three:
        best_match_indices = np.argsort(similarities)[::-1][:3]
        best_match_names = [file_names[i] for i in best_match_indices]
        best_match_scores = [similarities[i] for i in best_match_indices]
        return best_match_names, best_match_scores, similarities

    # Show similarity scores for all documents
    if verbose:
        for i, score in enumerate(similarities):
            print(f"Document {i}:")
            print(f"Content: {documents[i][:100]}...")  # Show a snippet of the document
            print(f"Similarity Score: {score:.4f}\n")

    # extract the name of the best matching document
    best_match_name = file_names[best_match_index]

    return best_match_name, similarities[best_match_index], similarities


# function to help print
def print_top_three_matches(best_match_names, best_match_scores):
    print("\nTop 3 matching clauses and their similarity scores:")
    for i in range(3):
        print(f"{i+1}. Clause: {best_match_names[i]}")
        print(f"   Similarity Score: {best_match_scores[i]:.4f}\n")


# ask the user to input a document
print("Please input a document path:")
query = input()

# find the best matching document
with open(query, "r", encoding="utf-8") as file:
    target_doc = file.read()

best_match_name, best_match_score, similarities = find_best_matching_document(
    target_doc, model, document_vectors, documents, top_three=True
)
print_top_three_matches(best_match_name, best_match_score)
