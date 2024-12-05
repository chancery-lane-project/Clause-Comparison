# TCLP x Faculty AI Fellowship December 20204 
## Detecting climate-aligned content in contracts and recommending TCLP content for contracts that are not already climate-aligned
![Alt text](readme_image.png)

### About the Project
The Chancery Lane Project was interested in exploring how AI could be used to increase organizational impact. This project included three sub-tasks: 
1. #### **Clause Recommender**  
   A **LegalBERT model**, paired with cosine similarity, analyzes contracts or legal documents to recommend the most relevant TCLP clauses.  

   **Output:** The top three most applicable clauses for the document.

3. #### **Clause Generation**  
   To create a synthetic database of contracts with TCLP-style content, an **OpenAI LLM** was fine-tuned on TCLP clauses.  

   **Key Highlights:**
   - Generated 1,800 novel clauses in TCLP style based on combinations of three keywords.
   - Keywords were derived from the most frequent terms in the original ~200 clauses.

4. #### **Clause Detection**
   The standout feature of the project, the **Clause Detector**, identifies the likelihood of TCLP or TCLP-inspired clauses (a.k.a. climate-aligned content) in a document. It uses a **TDIF vectorizer with logistic regression for binary classification**, applied on chunks of text. Then, these results are extrapolated to the entire document, combined with a threshold for **multiclass classification**. 

   **Performance:**  
   - Achieved **94% accuracy** on the test set.  
   - Enables efficient and precise metrics for TCLP to measure impact.

### Getting Started 

Follow these steps to set up the project and run it locally:

#### Prerequisites

Make sure you have the following installed:
- [Docker Desktop](https://www.docker.com/products/docker-desktop) (installed and running)
- Git (to clone the repository)

#### Setup Instructions

1. Clone the Repository: Open your terminal and run
```
`git clone https://github.com/chancery-lane-project/Clause-Comparison.git`
`cd Clause-Comparison`
```
Make sure you have a Personal Access Token set-up; instructions [here]([url](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)).

3. Download and Prepare the Model: Download the model and embeddings from [this link](https://drive.google.com/file/d/1sTpo9iOjhoCZ1qteLqry8jjezWTanSl_/view?usp=drive_link), unzip it, and place the files into the `Clause-Comparison/tclp/legalbert` folder to ensure the paths are correctly configured. If you are interested in having testing data or running some of the training notebooks, you can download the data from [this link](https://drive.google.com/drive/folders/1UJqd7kyTgziS1sDf67KQWfdg80hduY6w?usp=drive_link)

4. Run the Project with Docker: From the root directory of the project (`Clause-Comparison`), within your terminal:
```
`docker-compose up --build`
```
This will build and start the application. Make sure the Docker application is open and running. Then, you can navigate to the 'containers' tab in Docker and click one of the two links to see either the Clause Recommender or Clause Detector. 

You are now ready to explore the project's features!
