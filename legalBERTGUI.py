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

light_blue_text = "#e6f5ff"
mid_blue_text = "#b8e2ff"
navy = "#001f3f"
font = "Helvetica"


def create_rounded_button(
    parent, text, command, bg, fg, font=("Arial", 16), padding=10
):
    """
    Creates a rounded button using Canvas with text overlay.
    """
    # Create a canvas for the button background
    button_canvas = tk.Canvas(parent, bg=navy, highlightthickness=0)

    # Get text width to determine button size
    text_id = button_canvas.create_text(
        padding, padding, text=text, font=font, fill=fg, anchor="w"
    )
    text_bbox = button_canvas.bbox(text_id)
    text_width = text_bbox[2] - text_bbox[0] + padding * 2
    text_height = text_bbox[3] - text_bbox[1] + padding * 2

    # Draw a rounded rectangle
    radius = 15  # Radius for the rounded corners
    button_canvas.create_rounded_rect(
        x1=0, y1=0, x2=text_width, y2=text_height, radius=radius, fill=bg, outline=""
    )

    # Move text to the center of the rounded button
    button_canvas.coords(text_id, text_width / 2, text_height / 2)

    # Bind click event to run the command
    button_canvas.bind("<Button-1>", lambda e: command())
    button_canvas.itemconfig(text_id, tags=("button_text"))
    button_canvas.tag_bind("button_text", "<Button-1>", lambda e: command())

    # Set exact width and height, make canvas dynamically resizable
    button_canvas.config(width=text_width, height=text_height)

    return button_canvas


class TCLPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TCLP Clause Matcher")
        self.root.geometry("700x600")
        self.root.configure(bg=navy)

        # Configure style
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self.style.configure(
            "TButton",
            font=(font, 12),
            padding=6,
            background=light_blue_text,  # Light blue button color
            foreground=navy,
            relief="flat",
            anchor="center",
        )
        self.style.configure(
            "TLabel", font=(font, 12), background=navy, foreground=light_blue_text
        )  # Light blue text
        self.style.configure("TCombobox", font=(font, 12), padding=5)
        self.style.configure("TProgressbar", thickness=10)

        # Attributes for model, tokenizer, and data storage
        self.model = None
        self.tokenizer = None
        self.documents = []
        self.file_names = []
        self.embeddings = {}
        self.top_clause_text = ""
        self.contract_text = ""

        # Loading screen
        self.loading_screen = tk.Label(
            self.root,
            text="Preparing the application. This may take a few minutes the first time you open the app. After that, it will be instantaneous",
            font=(font, 12),
            wraplength=500,
            justify="center",
            bg=navy,
            fg=light_blue_text,
        )
        self.loading_screen.pack(pady=50)

        # Progress bar
        self.progress = ttk.Progressbar(
            self.root, orient="horizontal", mode="determinate", length=400
        )
        self.progress.pack(pady=20)

        # Start loading model in a separate thread
        self.load_model_thread = threading.Thread(target=self.load_model)
        self.load_model_thread.start()

    def load_model(self):
        local_model_dir = "./legalbert_model"
        embeddings_dir = "./legalbert_embeddings"
        self.progress["maximum"] = 6

        if not os.path.exists(local_model_dir):
            os.makedirs(local_model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained("casehold/legalbert")
            self.model = AutoModel.from_pretrained("casehold/legalbert")
            self.tokenizer.save_pretrained(local_model_dir)
            self.model.save_pretrained(local_model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
            self.model = AutoModel.from_pretrained(local_model_dir)

        self.progress["value"] += 1
        self.root.update_idletasks()

        folder_path = "England:Wales"
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    self.documents.append(file.read())
                    self.file_names.append(filename)

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

            self.progress["value"] += 1
            self.root.update_idletasks()

        self.loading_screen.pack_forget()
        self.progress.pack_forget()
        self.show_main_gui()

    def show_main_gui(self):
        self.banner = tk.Label(
            self.root,
            text="Remember these are just suggestions! Please review each clause for your specific needs.",
            font=(font, 14, "bold"),
            wraplength=600,
            fg="white",
            bg=navy,
        )
        self.banner.pack(pady=10)

        self.method_var = tk.StringVar()
        self.method_var.set("cls")
        method_label = ttk.Label(self.root, text="Select Embedding Method:")
        method_label.pack(pady=(10, 0))
        self.method_dropdown = ttk.Combobox(
            self.root,
            textvariable=self.method_var,
            values=["cls", "mean", "max", "concat", "specific"],
            state="readonly",
        )
        self.method_dropdown.pack(pady=10)

        self.browse_button = tk.Button(
            self.root,
            text="Browse File",
            command=self.browse_file,
            font=(font, 12),
            bg=light_blue_text,
            fg=navy,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            padx=10,
            pady=5,
        )
        self.browse_button.pack(pady=10)

        self.view_contract_button = tk.Button(
            self.root,
            text="View Contract Text",
            command=self.view_contract,
            state=tk.DISABLED,
            font=(font, 12),
            bg=light_blue_text,
            fg=navy,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            padx=10,
            pady=5,
        )
        self.view_contract_button.pack(pady=5)

        self.result_frame = tk.Frame(self.root, bg=navy)
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
            token_index = inputs["input_ids"].shape[1] - 1
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
        return filename.replace("_", " ").replace(".txt", "")

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                query = file.read()
            self.contract_text = query
            self.view_contract_button.config(state=tk.NORMAL)
            selected_method = self.method_var.get()
            self.display_results(query, selected_method)

    def display_results(self, query, method):
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        instruction_label = tk.Label(
            self.result_frame,
            text="Click on the name of the clauses below to see the full text of the top 3 matching clauses for your contract:",
            font=(font, 16, "bold"),
            fg=mid_blue_text,  # Slightly darker light blue
            bg=navy,
            wraplength=500,
        )
        instruction_label.pack(pady=5)

        best_match_names, best_match_scores = self.find_best_matching_clause(
            query, method
        )

        for i, (name, score) in enumerate(zip(best_match_names, best_match_scores), 1):
            score_frame = tk.Frame(self.result_frame, bg=navy, padx=10, pady=5)
            score_frame.pack(pady=5, fill="x")

            score_label = tk.Label(
                score_frame,
                text=f"Similarity Score: {score:.4f}",
                font=(font, 12, "bold"),
                fg=light_blue_text,
                bg=navy,
            )
            score_label.pack(side="top", anchor="w")

            clause_button = tk.Button(
                score_frame,
                text=name,
                font=(font, 16),
                bg=light_blue_text,
                fg=navy,
                borderwidth=0,
                relief="flat",
                command=lambda n=name: self.view_clause_text(n),
            )
            clause_button.pack(side="bottom", anchor="w", fill="x", padx=5, pady=5)

    def view_contract(self):
        self.show_text_in_window("Contract Text", self.contract_text)

    def view_clause_text(self, clause_name):
        clause_index = self.file_names.index(f"{clause_name.replace(' ', '_')}.txt")
        clause_text = self.documents[clause_index]
        self.show_text_in_window(f"Clause: {clause_name}", clause_text)

    def show_text_in_window(self, title, content):
        new_window = tk.Toplevel(self.root)
        new_window.title(title)
        text_widget = tk.Text(
            new_window, wrap=tk.WORD, font=(font, 12), bg=navy, fg=light_blue_text
        )
        text_widget.insert(tk.END, content)
        text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        text_widget.config(state=tk.DISABLED)


def main():
    nltk.download("punkt")
    root = tk.Tk()
    app = TCLPApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
