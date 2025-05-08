# utils.py
"""This is the utils file for the clause_recommender task."""
import os
import re
from difflib import get_close_matches

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_embeddings(method, embeddings_dir, documents, tokenizer, model):
    path = os.path.join(embeddings_dir, f"{method}_embeddings.npy")
    if os.path.exists(path):
        return np.load(path)
    else:
        embs = np.vstack([
            encode_text(doc, tokenizer, model, method)
            for doc in documents
        ])
        np.save(path, embs)
        return embs
    
def load_clauses(folder_path):
    documents = []
    file_names = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if (
            os.path.isfile(file_path)
            and filename.endswith(".txt")
            or filename.endswith(".html")
        ):
            with open(file_path, "r", encoding="utf-8") as file:
                documents.append(file.read())
                file_names.append(filename)

    return documents, file_names


def open_target(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def custom_stop_words():
    legal_words = [
        "clause",
        "agreement",
        "contract",
        "parties",
        "shall",
        "herein",
        "company",
        "date",
        "form",
        "party",
        "ii",
        "pursuant",
        "document",
        "term",
        "condition",
        "obligation",
        "rights",
        "include",
        "including",
        "one",
        "such",
        "set",
        "environment",
        "environmental",
        "jurisdiction",
        "jurisdictional",
        "class",
        "clause",
        "clauses"
    ]
    # Ensure all words are lowercase
    combined = ENGLISH_STOP_WORDS.union(w.lower() for w in legal_words)
    return list(combined)

def get_matching_clause(query_vector, document_vectors, clause_names):
    # Compute cosine similarities using a single call
    cosine_similarities = cosine_similarity(query_vector, document_vectors).flatten()

    # Retrieve the best match and its details
    most_similar_index = np.argmax(cosine_similarities)
    most_similar_score = cosine_similarities[most_similar_index]
    similarity_range = cosine_similarities.max() - cosine_similarities.min()
    most_similar_clause = clause_names[most_similar_index]

    return (
        most_similar_clause,
        most_similar_score,
        most_similar_index,
        cosine_similarities,
        similarity_range,
    )


def find_top_k(similarities, clause_names, k=3):
    best_match_indices = np.argsort(similarities)[::-1][:k]
    best_match_names = [clause_names[i] for i in best_match_indices]
    best_match_scores = [similarities[i] for i in best_match_indices]
    return best_match_names, best_match_scores, similarities


def print_similarities(most_similar_clause, most_similar_score, similarity_range):
    print(f"Most similar clause: {most_similar_clause}")
    print(f"Cosine similarity score: {most_similar_score:.2f}")
    print(f"Similarity range: {similarity_range:.2f}")


def output_feature_chart(vectorizer, X, most_similar_index):
    # Get the feature names (words) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Convert the document-term matrix to an array and isolate the most similar document and target doc
    X_array = X.toarray()
    target_vector = X_array[-1]  # Target document vector
    most_similar_vector = X_array[most_similar_index]  # Most similar document vector

    # Create DataFrames to easily view word frequencies
    target_df = pd.DataFrame({"word": feature_names, "target_frequency": target_vector})
    similar_df = pd.DataFrame(
        {"word": feature_names, "similar_frequency": most_similar_vector}
    )

    # Merge on words and filter for words that have non-zero counts in both documents
    merged_df = target_df.merge(similar_df, on="word")
    merged_df = merged_df[
        (merged_df["target_frequency"] > 0) & (merged_df["similar_frequency"] > 0)
    ]

    # Sort by combined frequency (or just display both frequencies)
    merged_df["total_frequency"] = (
        merged_df["target_frequency"] + merged_df["similar_frequency"]
    )
    merged_df = merged_df.sort_values(by="total_frequency", ascending=False)

    return merged_df


def process_similarity_df(similarities):
    # Output the target documents sorted by similarity score
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    similarity_df = pd.DataFrame(sorted_similarities, columns=["Document", "Clause"])

    # if the column contains three elements, split it into three columns
    if len(similarity_df["Clause"].iloc[0]) == 3:
        similarity_df[["Similarity Score", "Clause Text", "Top Words"]] = pd.DataFrame(
            similarity_df["Clause"].tolist(), index=similarity_df.index
        )
    else:
        similarity_df[["Similarity Score", "Clause Text"]] = pd.DataFrame(
            similarity_df["Clause"].tolist(), index=similarity_df.index
        )

    similarity_df = similarity_df.drop(columns=["Clause"])

    return similarity_df


def unique_clause_counter(similarity_df):
    # number of unique clauses over 50% similarity
    unique_clauses = similarity_df[similarity_df["Similarity Score"] > 0.5]
    unique_clause_list = unique_clauses["Clause Text"].tolist()
    unique_clause_list = pd.DataFrame(unique_clause_list, columns=["Clause"])
    unique_clause_list["Count"] = 1
    unique_clause_list = unique_clause_list.groupby("Clause").count().reset_index()
    unique_clause_list = unique_clause_list.sort_values(by="Count", ascending=False)

    return unique_clause_list


def graph_ranges(similarity_ranges):
    # plot the range of similarity differences
    plt.hist(similarity_ranges, bins=20)
    plt.xlabel("Difference in Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Distribution of Cosine Similarity Differences")
    plt.show()


# Helper Functions for Pooling
def cls_pooling(outputs):
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def mean_pooling(outputs):
    embeddings = outputs.last_hidden_state
    return embeddings.mean(dim=1).cpu().numpy()


def max_pooling(outputs):
    embeddings = outputs.last_hidden_state
    return embeddings.max(dim=1).values.cpu().numpy()


def concat_pooling(outputs):
    embeddings = outputs.last_hidden_state
    mean_pooling = embeddings.mean(dim=1)
    max_pooling = embeddings.max(dim=1).values
    return torch.cat((mean_pooling, max_pooling), dim=1).cpu().numpy()


def specific_token_pooling(outputs, token_index=None):
    # Determine the token index for the last token if not specified
    if token_index is None:
        token_index = outputs.last_hidden_state.size(1) - 1  # Get the last token index
    return outputs.last_hidden_state[:, token_index, :].cpu().numpy()


def encode_text(text, tokenizer, model, method="cls", token_index=None):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)

    if method == "cls":
        return cls_pooling(outputs)
    elif method == "mean":
        return mean_pooling(outputs)
    elif method == "max":
        return max_pooling(outputs)
    elif method == "concat":
        return concat_pooling(outputs)
    elif method == "specific":
        return specific_token_pooling(outputs, token_index)
    else:
        raise ValueError(
            "Invalid method. Choose from 'cls', 'mean', 'max', 'concat', 'specific'."
        )


def encode_all(clauses, target_doc, tokenizer, model, method="cls"):
    # Encode all clauses using the specified pooling method
    clause_embeddings = np.vstack(
        [encode_text(clause, tokenizer, model, method) for clause in clauses]
    )

    # Encode the target document
    target_doc_embedding = encode_text(target_doc, tokenizer, model, method)

    return clause_embeddings, target_doc_embedding


def find_top_similar_bow(target_doc, documents, file_names, similarity_threshold=0.15, k=30):
    custom_stop_words_list = custom_stop_words()
    vectorizer = CountVectorizer(stop_words=custom_stop_words_list, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
    all_documents = documents + [target_doc]
    X = vectorizer.fit_transform(all_documents)
    document_vectors = X[:-1]
    query_vector = X[-1:]

    # Call helper method to get the most similar clause
    (
        _,
        _,
        most_similar_index,
        similarities,
        _,
    ) = get_matching_clause(query_vector, document_vectors, file_names)

    # Find the top three matching clauses
    best_match_names, best_match_scores, similarities = find_top_k(
        similarities, file_names, k = k
    )
    merged_df = output_feature_chart(vectorizer, X, most_similar_index)

    return {
    "Top_Matches": best_match_names,
    "Scores": best_match_scores,
    "Documents": [documents[file_names.index(name)] for name in best_match_names],
    "Feature_Chart": merged_df,
    }
    
def get_embedding_matches_subset(
    query_text, 
    documents_subset, 
    names_subset, 
    tokenizer, 
    model, 
    method="cls", 
    k =3
):
    # Embed the top-K candidate clauses
    subset_embeddings = np.vstack([
        encode_text(doc, tokenizer, model, method)
        for doc in documents_subset
    ])
    # Embed the query
    query_embedding = encode_text(query_text, tokenizer, model, method).reshape(1, -1)

    # Compute cosine similarity
    _, _, _, similarities, _ = get_matching_clause(query_embedding, subset_embeddings, names_subset)

    # Get top 3 (or however many you want)
    top_names, top_scores, _ = find_top_k(similarities, names_subset, k=k)
    top_texts = [documents_subset[names_subset.index(name)] for name in top_names]

    return top_names, top_scores, top_texts

def parse_clause_boxes_to_df(clause_boxes):
    data = []

    for box in clause_boxes:
        soup = BeautifulSoup(box, "html.parser")
        name = soup.find("p", class_="childs-name").get_text(strip=True) if soup.find("p", class_="childs-name") else None
        title = soup.find("h4").get_text(strip=True) if soup.find("h4") else None
        excerpt = soup.find("p", class_="excerpt").get_text(strip=True) if soup.find("p", class_="excerpt") else None

        # Initialize
        jurisdiction, updated = None, None
        
        meta_data = soup.find_all("p", class_="meta-data")
        for md in meta_data:
            leadin = md.find("span", class_="cfc-leadin")
            value = md.find("span", class_="cfc-taxonomy")
            if leadin and value:
                leadin_text = leadin.get_text(strip=True)
                value_text = value.get_text(strip=True)
                if "Jurisdiction" in leadin_text:
                    jurisdiction = value_text
                elif "Updated" in leadin_text:
                    updated = value_text

        url_tag = soup.find("a", class_="hot-spot")
        url = url_tag['href'] if url_tag and url_tag.has_attr('href') else None

        data.append({
            "Name": name,
            "Title": title,
            "Excerpt": excerpt,
            "Jurisdiction": jurisdiction,
            "Updated": updated,
            "URL": url
        })

    return pd.DataFrame(data)

def clean_string(s):
    s = s.lower()
    s = re.sub(r'[\W_]+', '_', s)  # Replace non-alphanumeric characters with underscores
    s = re.sub(r'_+', '_', s)  # Collapse multiple underscores
    return s.strip('_')


def attach_documents(df, documents, file_names):
    # Preprocess file_names
    file_map = {clean_string(name.replace('.txt', '')): (name, doc) for name, doc in zip(file_names, documents)}
    
    records = []
    missing_titles = []
    
    for idx, row in df.iterrows():
        title = row['Title']
        cleaned_title = clean_string(title)
        possible_matches = get_close_matches(cleaned_title, file_map.keys(), n=1, cutoff=0.6)
        
        if possible_matches:
            best_match = possible_matches[0]
            filename, text = file_map[best_match]
            new_row = row.copy()
            new_row['Matched Filename'] = filename
            new_row['Document'] = text
            records.append(new_row)
        else:
            missing_titles.append(title)
    
    # Build new df only with matched records
    result_df = pd.DataFrame(records)
    
    if missing_titles:
        print("\n⚠️ Titles with no matching document:")
        for title in missing_titles:
            print(f" - {title}")
    
    return result_df
        
def rebuild_documents(df):
    filenames = []
    documents = []
    
    for idx, row in df.iterrows():
        filename = str(row['Title']).strip()  # Force safe string
        excerpt = str(row['Excerpt'] or "").strip()
        document = str(row['Document'] or "").strip()
        
        combined_text = f"{excerpt}\n\n{document}".strip()
        
        filenames.append(filename)
        documents.append(combined_text)
    
    return filenames, documents


