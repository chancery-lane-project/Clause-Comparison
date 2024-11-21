import os
import random
import argparse
from clause_matcher import LegalBERTMatcher

# initialize paths and matcher
matcher = LegalBERTMatcher()
clauses_folder_path = "cleaned_clauses"
modified_folder_path = "modified_synthetic_data"
untouched_folder_path = "untouched_synthetic_data"
os.makedirs(modified_folder_path, exist_ok=True)
os.makedirs(untouched_folder_path, exist_ok=True)


def insert_randomly(contract_path, clauses):
    with open(contract_path, "r", encoding="utf-8") as contract_file:
        text = contract_file.read()
    to_insert = random.choice(clauses)[0]
    clause_path = os.path.join(
        clauses_folder_path, to_insert.replace(" ", "_") + ".txt"
    )
    with open(clause_path, "r", encoding="utf-8") as clause_file:
        clause_content = clause_file.read()

    lines = text.splitlines()
    clause_lines = clause_content.splitlines()
    paragraph_breaks = [
        i
        for i in range(1, len(lines) - 1)
        if lines[i].strip() == "" and lines[i - 1].strip() and lines[i + 1].strip()
    ]
    index = random.choice(paragraph_breaks) if paragraph_breaks else len(lines)

    # create labeled lines
    labeled_text = [(line, "0") for line in lines[:index]]
    labeled_text += [("", "0")]
    labeled_text += [(line, "1") for line in clause_lines]
    labeled_text += [("", "0")]
    labeled_text += [(line, "0") for line in lines[index:]]

    return labeled_text


def label_0s(contract_path):
    with open(contract_path, "r", encoding="utf-8") as contract_file:
        text = contract_file.read()
    lines = text.splitlines()
    labeled_text = [(line, "0") for line in lines]
    return labeled_text


def get_unique_filename(output_folder, filename):
    """Generate a unique filename by appending a suffix if the file already exists."""
    base_name, extension = os.path.splitext(filename)
    counter = 1
    unique_filename = filename

    while os.path.exists(os.path.join(output_folder, unique_filename)):
        unique_filename = f"{base_name}_{counter}{extension}"
        counter += 1

    return unique_filename


def process_contracts_folder(
    contracts_folder_path,
    clauses_folder_path,
    modified_folder,
    untouched_folder,
    modification_chance=0.5,
):
    os.makedirs(modified_folder, exist_ok=True)
    os.makedirs(untouched_folder, exist_ok=True)
    """Processes each contract file in all subfolders, randomly modifying 50% of them."""
    for root, _, files in os.walk(contracts_folder_path):
        for contract_filename in files:
            if not contract_filename.lower().endswith(".txt"):
                print(f"Skipping non-text file: {contract_filename}")
                continue

            print(f"Processing: {contract_filename}")
            contract_path = os.path.join(root, contract_filename)
            if not os.path.isfile(contract_path):
                continue

            # 50% chance to modify the contract
            modify_contract = random.random() < modification_chance
            output_folder = modified_folder if modify_contract else untouched_folder
            output_contract_path = os.path.join(output_folder, contract_filename)

            # Ensure a unique output file name
            output_contract_path = os.path.join(
                output_folder, get_unique_filename(output_folder, contract_filename)
            )

            if modify_contract:
                try:
                    top_clauses = matcher.match_clauses(
                        contract_path, clauses_folder_path, method="cls"
                    )
                    labeled_content = insert_randomly(contract_path, top_clauses)
                    modified_content = "\n".join(
                        f"{label} {line}" for line, label in labeled_content
                    )
                    with open(
                        output_contract_path, "w", encoding="utf-8"
                    ) as new_contract_file:
                        new_contract_file.write(modified_content)
                except UnicodeDecodeError:
                    print(f"Error: UTF-8 decoding failed for file: {contract_path}")
            else:
                zeroed_content = label_0s(contract_path)
                zero_label_content = "\n".join(
                    f"{label} {line}" for line, label in zeroed_content
                )
                try:
                    with open(
                        output_contract_path, "w", encoding="utf-8"
                    ) as new_contract_file:
                        new_contract_file.write(zero_label_content)
                except UnicodeDecodeError:
                    print(f"Error: UTF-8 decoding failed for file: {contract_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a folder of contracts, randomly modifying 50% of them with clauses."
    )
    parser.add_argument(
        "contracts_folder",
        nargs="?",
        help="Path to the folder containing contract files",
    )
    args = parser.parse_args()

    if not args.contracts_folder:
        args.contracts_folder = input(
            "Please provide a contracts folder path: "
        ).strip()

    if not args.contracts_folder:
        print("Error: Contracts folder path is required.")
    else:
        process_contracts_folder(
            args.contracts_folder,
            clauses_folder_path,
            modified_folder_path,
            untouched_folder_path,
            modification_chance=0.5,
        )
