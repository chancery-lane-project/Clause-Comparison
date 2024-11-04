# bag of words with GUI

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load documents from the folder
def load_documents(folder_path):
    documents = []
    file_names = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                documents.append(file.read())
                file_names.append(filename)

    return documents, file_names


# Find the top 3 most similar documents and return relevant data
def find_top_similar(target_doc, documents, file_names, similarity_threshold=0.15):
    custom_stop_words = [
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

    vectorizer = CountVectorizer(stop_words="english")
    combined_stop_words = list(
        set(vectorizer.get_stop_words()).union(custom_stop_words)
    )
    vectorizer = CountVectorizer(stop_words=combined_stop_words)

    all_documents = documents + [target_doc]
    X = vectorizer.fit_transform(all_documents)

    cosine_similarities = cosine_similarity(X[-1:], X[:-1])
    top_indices = np.argsort(cosine_similarities[0])[::-1][:3]
    top_scores = cosine_similarities[0][top_indices]

    if all(score > similarity_threshold for score in top_scores):
        return (
            [file_names[i] for i in top_indices],
            top_scores,
            [documents[i] for i in top_indices],
            vectorizer,
            X,
            top_indices,
        )
    else:
        return [], top_scores, [], vectorizer, X, top_indices


# Generate the feature chart for overlapping words
def output_feature_chart(vectorizer, X, most_similar_index):
    feature_names = vectorizer.get_feature_names_out()
    X_array = X.toarray()
    target_vector = X_array[-1]
    most_similar_vector = X_array[most_similar_index]

    target_df = pd.DataFrame({"word": feature_names, "target_frequency": target_vector})
    similar_df = pd.DataFrame(
        {"word": feature_names, "similar_frequency": most_similar_vector}
    )

    merged_df = target_df.merge(similar_df, on="word")
    merged_df = merged_df[
        (merged_df["target_frequency"] > 0) & (merged_df["similar_frequency"] > 0)
    ]
    merged_df["total_frequency"] = (
        merged_df["target_frequency"] + merged_df["similar_frequency"]
    )
    merged_df = merged_df.sort_values(by="total_frequency", ascending=False)

    return merged_df


# GUI Application
class SimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Similarity Checker")
        self.root.geometry("600x700")

        self.documents, self.file_names = load_documents("England:Wales")
        self.merged_df = None
        self.top_clause_text = ""
        self.contract_text = ""

        # UI Elements
        self.label = ttk.Label(
            self.root, text="Select a document to compare", font=("Arial", 14)
        )
        self.label.pack(pady=20)

        self.browse_button = ttk.Button(
            self.root, text="Browse File", command=self.browse_file
        )
        self.browse_button.pack(pady=10)

        self.result_text = tk.Text(
            self.root, wrap=tk.WORD, height=10, font=("Arial", 12)
        )
        self.result_text.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        self.view_chart_button = ttk.Button(
            self.root,
            text="View Feature Chart",
            command=self.view_feature_chart,
            state=tk.DISABLED,
        )
        self.view_chart_button.pack(pady=5)

        self.view_clause_button = ttk.Button(
            self.root,
            text="View Top Clause",
            command=self.view_top_clause,
            state=tk.DISABLED,
        )
        self.view_clause_button.pack(pady=5)

        self.view_contract_button = ttk.Button(
            self.root,
            text="View Uploaded Document",
            command=self.view_contract,
            state=tk.DISABLED,
        )
        self.view_contract_button.pack(pady=5)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                self.contract_text = file.read()
            self.display_results(self.contract_text)

    def display_results(self, target_doc):
        top_names, top_scores, top_texts, vectorizer, X, top_indices = find_top_similar(
            target_doc, self.documents, self.file_names
        )

        self.result_text.delete(1.0, tk.END)

        if top_names:
            self.top_clause_text = top_texts[0]
            self.result_text.insert(
                tk.END, "Top 3 matching clauses and their similarity scores:\n\n"
            )
            for i, (name, score) in enumerate(zip(top_names, top_scores), 1):
                self.result_text.insert(tk.END, f"{i}. Clause: {name}\n")
                self.result_text.insert(tk.END, f"   Similarity Score: {score:.4f}\n\n")

            self.merged_df = output_feature_chart(vectorizer, X, top_indices[0])
            self.view_chart_button.config(state=tk.NORMAL)
            self.view_clause_button.config(state=tk.NORMAL)
            self.view_contract_button.config(state=tk.NORMAL)
        else:
            self.result_text.insert(
                tk.END,
                "Sorry! It looks like there aren't any good matches for your contract.\n",
            )
            self.view_chart_button.config(state=tk.DISABLED)
            self.view_clause_button.config(state=tk.DISABLED)
            self.view_contract_button.config(state=tk.DISABLED)

    def view_feature_chart(self):
        if self.merged_df is not None:
            new_window = tk.Toplevel(self.root)
            new_window.title("Feature Chart")
            text_widget = tk.Text(new_window, wrap=tk.WORD, font=("Arial", 12))
            text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

            text_widget.insert(
                tk.END, "Top overlapping words contributing to similarity:\n\n"
            )
            for _, row in self.merged_df.head(10).iterrows():
                text_widget.insert(
                    tk.END,
                    f"{row['word']}: {row['target_frequency']} (target), {row['similar_frequency']} (similar)\n",
                )
            text_widget.config(state=tk.DISABLED)

    def view_top_clause(self):
        self.show_text_in_window("Top Clause Text", self.top_clause_text)

    def view_contract(self):
        self.show_text_in_window("Uploaded Document Text", self.contract_text)

    def show_text_in_window(self, title, content):
        new_window = tk.Toplevel(self.root)
        new_window.title(title)
        text_widget = tk.Text(new_window, wrap=tk.WORD, font=("Arial", 12))
        text_widget.insert(tk.END, content)
        text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        text_widget.config(state=tk.DISABLED)


# Main function to run the GUI
def main():
    root = tk.Tk()
    app = SimilarityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
