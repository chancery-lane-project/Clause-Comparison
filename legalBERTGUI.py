# legalBERTGUI.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from transformers import AutoTokenizer, AutoModel
import torch
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import threading
import nltk


class TCLPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TCLP Clause Matcher")
        self.root.geometry("700x600")

        # Attributes for model, tokenizer, and data storage
        self.model = None
        self.tokenizer = None
        self.documents = []
        self.file_names = []
        self.embeddings = {}  # Dictionary to store embeddings for each method
        self.top_clause_text = ""
        self.contract_text = ""

        # Initialize a loading screen
        self.loading_screen = tk.Label(
            self.root,
            text="We are preparing your application! This will take a little bit of time the first time you use it but will be very quick after that.",
            font=("Arial", 12),
            wraplength=500,
            justify="center",
        )
        self.loading_screen.pack(pady=100)

        # Add a progress bar to the loading screen
        self.progress = ttk.Progressbar(
            self.root, orient="horizontal", mode="determinate", length=300
        )
        self.progress.pack(pady=10)

        # Start loading model and embeddings in a separate thread
        self.load_model_thread = threading.Thread(target=self.load_model)
        self.load_model_thread.start()

    def load_model(self):
        # Specify directories for model and embeddings
        local_model_dir = "./legalbert_model"
        embeddings_dir = "./legalbert_embeddings"

        # Set progress bar maximum (1 for model load + 5 for embedding methods)
        self.progress["maximum"] = 6

        # Load or download the model and tokenizer
        if not os.path.exists(local_model_dir):
            os.makedirs(local_model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained("casehold/legalbert")
            self.model = AutoModel.from_pretrained("casehold/legalbert")
            self.tokenizer.save_pretrained(local_model_dir)
            self.model.save_pretrained(local_model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
            self.model = AutoModel.from_pretrained(local_model_dir)

        # Update progress after model loading
        self.progress["value"] += 1
        self.root.update_idletasks()

        # Load documents
        folder_path = "England:Wales"
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    self.documents.append(file.read())
                    self.file_names.append(filename)

        # Precompute and save embeddings for each method if not already saved
        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir)

        for method in ["cls", "mean", "max", "concat", "specific"]:
            embedding_path = os.path.join(embeddings_dir, f"{method}_embeddings.npy")
            if os.path.exists(embedding_path):
                self.embeddings[method] = np.load(embedding_path)
            else:
                self.embeddings[method] = np.vstack(
                    [self.encode_text(doc, method) for doc in self.documents]
                )
                np.save(embedding_path, self.embeddings[method])

            # Update progress after each embedding computation
            self.progress["value"] += 1
            self.root.update_idletasks()

        # Switch to main GUI once loading is complete
        self.loading_screen.pack_forget()
        self.progress.pack_forget()
        self.show_main_gui()

    def show_main_gui(self):
        # Add banner message at the top of the main GUI with larger font
        self.banner = tk.Label(
            self.root,
            text="Remember, these are suggestions! Please carefully review the clauses and see if they would be applicable for your case.",
            font=("Arial", 14, "bold"),  # Increased font size for visibility
            wraplength=600,
            fg="red",
        )
        self.banner.pack(pady=5)

        # Embedding Method Dropdown
        self.method_var = tk.StringVar()
        self.method_var.set("cls")  # default method
        self.method_dropdown = ttk.Combobox(
            self.root,
            textvariable=self.method_var,
            values=["cls", "mean", "max", "concat", "specific"],
        )
        self.method_dropdown.pack(pady=10)

        # Button to Browse for Contract File
        self.browse_button = ttk.Button(
            self.root, text="Browse File", command=self.browse_file
        )
        self.browse_button.pack(pady=10)

        # Button to View Contract Text after uploading
        self.view_contract_button = ttk.Button(
            self.root,
            text="View Contract Text",
            command=self.view_contract,
            state=tk.DISABLED,
        )
        self.view_contract_button.pack(pady=5)

        # Frame to display results as buttons
        self.result_frame = tk.Frame(self.root)
        self.result_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

    def encode_text(self, text, method="cls"):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        if method == "cls":
            return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        elif method == "mean":
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        elif method == "max":
            return outputs.last_hidden_state.max(dim=1).values.cpu().numpy().flatten()
        elif method == "concat":
            mean_pooling = outputs.last_hidden_state.mean(dim=1)
            max_pooling = outputs.last_hidden_state.max(dim=1).values
            return torch.cat((mean_pooling, max_pooling), dim=1).cpu().numpy().flatten()
        elif method == "specific":
            token_index = inputs["input_ids"].shape[1] - 1  # last token index
            return outputs.last_hidden_state[:, token_index, :].cpu().numpy().flatten()

    def find_best_matching_clause(self, query, method):
        clause_embeddings = self.embeddings[method]
        query_embedding = self.encode_text(query, method).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, clause_embeddings).flatten()

        best_match_indices = np.argsort(similarities)[::-1][:3]
        best_match_names = [
            self.format_clause_name(self.file_names[i]) for i in best_match_indices
        ]
        best_match_scores = [similarities[i] for i in best_match_indices]

        return best_match_names, best_match_scores

    def format_clause_name(self, filename):
        """Format clause name by replacing underscores with spaces and removing the .txt extension."""
        return filename.replace("_", " ").replace(".txt", "")

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                query = file.read()
            self.contract_text = query
            self.view_contract_button.config(
                state=tk.NORMAL
            )  # Enable "View Contract" button
            selected_method = self.method_var.get()
            self.display_results(query, selected_method)

    def display_results(self, query, method):
        # Clear previous results
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        # Add instruction label for the results
        instruction_label = tk.Label(
            self.result_frame,
            text="For your contract, the following three clauses match best. Click on the name of the contract to view it in full:",
            font=("Arial", 16),
            fg="blue",
            bg="white",
            wraplength=500,
        )
        instruction_label.pack(pady=5)

        # Display results as buttons
        best_match_names, best_match_scores = self.find_best_matching_clause(
            query, method
        )
        if all(score <= 0.5 for score in best_match_scores):
            no_match_label = tk.Label(
                self.result_frame,
                text="Sorry! It looks like there aren't any good matches for your contract.",
                font=("Arial", 12),
            )
            no_match_label.pack(pady=5)
            return

        for i, (name, score) in enumerate(zip(best_match_names, best_match_scores), 1):
            clause_button = tk.Button(
                self.result_frame,
                text=f"{i}. {name} - Similarity Score: {score:.4f}",
                font=("Arial", 14),
                command=lambda n=name: self.view_clause_text(n),
            )
            clause_button.pack(pady=5, fill="x")

    def view_contract(self):
        """Open the contract text in a new window."""
        self.show_text_in_window("Contract Text", self.contract_text)

    def view_clause_text(self, clause_name):
        clause_index = self.file_names.index(f"{clause_name.replace(' ', '_')}.txt")
        clause_text = self.documents[clause_index]

        self.show_text_in_window(f"Clause: {clause_name}", clause_text)

    def show_text_in_window(self, title, content):
        new_window = tk.Toplevel(self.root)
        new_window.title(title)
        text_widget = tk.Text(new_window, wrap=tk.WORD, font=("Arial", 12))
        text_widget.insert(tk.END, content)
        text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        text_widget.config(state=tk.DISABLED)


# Main function to run the GUI
def main():
    nltk.download("punkt")  # Ensure nltk is ready
    root = tk.Tk()
    app = TCLPApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
