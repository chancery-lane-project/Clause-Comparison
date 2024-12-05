# TCLP x Faculty AI Fellowship December 20204 
## Detecting climate-aligned content in contracts and recommending TCLP content for contracts that are not already climate-aligned
![Alt text](readme_image.png)

### About the Project
The Chancery Lane Project was interested in exploring how AI could be used to increase organizational impact. This project included three sub-tasks: 
1. **Clause Recommender**  
   A **LegalBERT model**, paired with cosine similarity, analyzes contracts or legal documents to recommend the most relevant TCLP clauses.  
   **Output:** The top three most applicable clauses for the document.

2. **Clause Generation**  
   To create a synthetic database of contracts with TCLP-style content, an **OpenAI LLM** was fine-tuned on TCLP clauses.  
   **Key Highlights:**
   - Generated 1,800 novel clauses in TCLP style based on combinations of three keywords.
   - Keywords were derived from the most frequent terms in the original ~200 clauses.

3. **Clause Detection**
   The standout feature of the project, the **Clause Detector**, identifies the likelihood of TCLP or TCLP-inspired clauses (a.k.a. climate-aligned content) in a document. It uses a **TDIF vectorizer with logistic regression for binary classification**, applied on chunks of text. Then, these results are extrapolated to the entire document, combined with a threshold for **multiclass classification**. 
   **Performance:**  
   - Achieved **94% accuracy** on the test set.  
   - Enables efficient and precise metrics for TCLP to measure impact.
