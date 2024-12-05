# TCLP x Faculty AI Fellowship December 20204 
## Detecting climate-aligned content in contracts and recommending TCLP content for contracts that are not already climate-aligned
![Alt text](readme_image.png)

## About the Project
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

## Getting Started 

Follow these steps to set up the project and run it locally:

### Prerequisites

Make sure you have the following installed:
- [Docker Desktop](https://www.docker.com/products/docker-desktop) (installed and running)
- Git (to clone the repository)

### Setup Instructions

#### Clone the Repository
Open your terminal and run:
```
`git clone https://github.com/chancery-lane-project/Clause-Comparison.git`
`cd Clause-Comparison`
```
Make sure you have a Personal Access Token set-up; instructions [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

#### Download and Prepare the Model
Download the model and embeddings from [this link](https://drive.google.com/file/d/1sTpo9iOjhoCZ1qteLqry8jjezWTanSl_/view?usp=drive_link), unzip it, and place the files into the `Clause-Comparison/tclp/legalbert` folder to ensure the paths are correctly configured. If you are interested in having testing data or running some of the training notebooks, you can download the data from [this link](https://drive.google.com/drive/folders/1UJqd7kyTgziS1sDf67KQWfdg80hduY6w?usp=drive_link). Place those files in 'Clause-Comparison/tclp/data'. The baseline data (including the clauses themselves) is already included for you in this repo, so it will be installed locally when you clone. 

#### Run the Project with Docker
From the root directory of the project (`Clause-Comparison`), within your terminal:
```
`docker-compose up --build`
```
This will build and start the application. Make sure the Docker application is open and running. Then, you can navigate to the 'containers' tab in Docker and click one of the two links to see either the Clause Recommender or Clause Detector. 

You are now ready to explore the project's features!

## Data and Model Access
Discussed in the previous section, in case you missed it, repeated here for comprehensiveness. 

### Access All Data 
[Google drive folder](https://drive.google.com/drive/folders/1UJqd7kyTgziS1sDf67KQWfdg80hduY6w?usp=drive_link) 
Download this and place it in Clause-Comparison/tclp/data

### Access LegalBERT model and embeddings 
[Google drive folder](https://drive.google.com/file/d/1sTpo9iOjhoCZ1qteLqry8jjezWTanSl_/view?usp=drive_link)
Download this and place it in Clause-Comparison/tclp/legalbert

## Usage
If you are **interested in using the frontend applications**, those will immediately be launched on a local server once you build the docker. They are designed to have a **simple user inteface**, easily intuited by non-technical and technical users alike. 

If you wish to explore the code in more depth and desire further ideas about exploration, please refer to the in-depth tour of the repo just below. 

## Repo Tour 
If you are more technically inclined, and want to understand the backend of these applications and the repository structure, this section is for you. 

### Landing Page

- **[`tclp/`](tclp)**: This is the main folder where most of the project code and data are located. Any downloaded models or embeddings should be placed in this directory for the project to function correctly. Think of it like an src file in other projects.
- **[`.dockerignore`](.dockerignore) and [`.gitignore`](.gitignore)**: These files define which files or folders should be excluded from Docker images and Git commits, respectively. These are important for discluding large stores of data from this repo. 
- **[`Dockerfile`](Dockerfile)**: Contains the instructions to build the Docker container for the project.
- **[`docker-compose.yaml`](docker-compose.yaml)**: Simplifies the setup of Docker services. It ensures the application and its dependencies are properly configured and running.
- **[`poetry.lock`](poetry.lock) and [`pyproject.toml`](pyproject.toml)**: Used for managing Python dependencies via Poetry. These files ensure consistent dependency versions and mean the user can easily install the required environment. 
- **[`README.md`](README.md)**: This document! 
- **[`readme_image.png`](readme_image.png)**: The opening image of this document.

### tclp 
Inside this source document there are further files and sub-folders. 

- **[`LLMs/`](tclp/LLMs)**: Contains pre-trained or fine-tuned language models used in the project, like LegalBERT or other custom models.
- **[`clause_detector/`](tclp/clause_detector)**: The logic and scripts for identifying whether a document contains TCLP-inspired or climate-aligned clauses.
- **[`clause_recommender/`](tclp/clause_recommender)**: Code responsible for suggesting the most relevant TCLP clauses for a given legal document.
- **[`data/`](tclp/data/)**: A directory for datasets or inputs required for testing, training, or running the project.
- **[`xml_parse.py`](tclp/xml_parse.py)**: This script likely handles XML-based contracts or clauses, helping preprocess or extract necessary information.
- **`__pycache__/`**: A Python-generated folder that caches compiled files for faster execution.
- **`.DS_Store`**: A macOS-specific file that stores folder metadata. This can be ignored or deleted as it serves no purpose for the project.
- **`__init__.py`**: Allows Python to treat this folder as a package, enabling imports from this directory in other scripts.

This structure organizes the project into distinct components for clause detection, recommendation, and general utilities, making it easy to navigate and extend the project.

