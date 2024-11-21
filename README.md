# Clause-Comparison

## Description of Content

The **xml_parser.py file** takes a messy XML file of all TCLP's website content and outputs a folder of text files containing only clauses, glossary terms, and guides. **A warning:** this was created based on the website architecture in October 2024 and if it ever changes dramatically (i.e. the name of "glossary terms" is changed), it will also need to be updated. Content uploaded in the same way as it currently has been will be fine. This can be used to re-pull content in the event that TCLP wants to retrain the model based on updated data to their website or for **any other reason TCLP might want to have txt files of all their content**. The user will have needed to downolad the XML file of all content and saved it to the folder where they intend to run this script.

The **determine_dialect.py file** is a tool to detect the predominant dialect (British or American English) in text files, specifically for legal contracts or similar documents. This script operates by scanning text for a comprehensive list of British and American word variants and place names, comparing occurrences to determine the dialect. It allows the user to check a single file or a folder of files. If a directory is selected, it will analyze each .txt file, automatically relocating those in British English to a separate British_Contracts subfolder for easy organization.

The **doc2vec** folder includes the TCLP clauses that pertain to England/Wales as well as glossary terms and guides. It also contains two scripts: one with a GUI and one without. Both match the user's uploaded document to the content it most matches from TCLP.

- **Notes**: This would be made better if it can scrape clauses from the TCLP website in real time, not needing to have them stored locally. It would also be better if the model did not train inside the script, and was instead loaded in.
- **bowGUI.py** and **legalBERTGUI.py** are similar to doc2vec but allows the user to play around with the bag of words and legal BERT approaches, respectively. Still needs the England/Wales folder although BOW does not train a model internally so it is improved in that way. legalBERT takes a while to load the pre-trained model and embed the clauses but the user only has to do that the first time they open the application; they are then stored locally. **encryptGUI.py** is largely the same as legalBERTGUI, except that it includes simple encryption.


**utils.py** is relevant to legalBERTGUI, bowGUI, and encryptGUI. It has shared helper functions to improve modularity and will need to be imported to each of them for the code to properly run.

**back_to_basics.ipynb** is the Jupyter Notebook for clause identification in contracts. It takes the synthetically generated data (users can make their own synthetic database using insert_clause.py) and identifies, on a chunk and contract level, the presence of clauses. To use it, all you need is folders with modified contracts (including clauses), unmodified clauses (not including clauses), and the clauses themselves.

**clause_matcher.py** is a class which encompasses the output of Task 2, allowing a user to "from clause_matcher import LegalBERTMatcher" and then use it as follows:


