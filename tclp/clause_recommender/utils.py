# utils.py
"""This is the utils file for the clause_recommender task."""
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch


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
    return [
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
    ]


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


def find_top_three(similarities, clause_names):
    best_match_indices = np.argsort(similarities)[::-1][:3]
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


def find_top_similar_bow(target_doc, documents, file_names, similarity_threshold=0.15):
    custom_stop_words_list = custom_stop_words()
    vectorizer = CountVectorizer(stop_words=custom_stop_words_list)
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
    best_match_names, best_match_scores, similarities = find_top_three(
        similarities, file_names
    )
    merged_df = output_feature_chart(vectorizer, X, most_similar_index)

    # Check if top matches meet similarity threshold
    if all(score > similarity_threshold for score in best_match_scores):
        return {
            "Top_Matches": best_match_names,
            "Scores": best_match_scores,
            "Documents": [
                documents[file_names.index(name)] for name in best_match_names
            ],
            "Feature_Chart": merged_df,
        }
    else:
        return {
            "Top_Matches": [],
            "Scores": [],
            "Documents": [],
            "Feature_Chart": merged_df,
        }
