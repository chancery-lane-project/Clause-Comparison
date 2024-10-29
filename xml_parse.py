# Necessary imports
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import os

# setting the stage with necessary variables and namespaces

# NOTE: The XML file must be downloaded from the WordPress site and saved in the same directory as this script
# Please name it "thechancerylaneproject.WordPress.xml" or update the file name in the code below
tree = ET.parse("thechancerylaneproject.WordPress.xml")

root = tree.getroot()
content_namespace = "{http://purl.org/rss/1.0/modules/content/}"
wp_namespace = "{http://wordpress.org/export/1.2/}"

# make a new directory to store the text files
output_dir = "tclp_content"
os.makedirs(output_dir, exist_ok=True)

# parsing the XML file and extracting the content; this code will extract clauses, guides, and glossary terms
target_post_types = {"clause", "guide", "glossary-term"}
counter = 0

for item in root.findall("channel/item"):
    # find the post type for each item
    post_type_element = item.find(f"{wp_namespace}post_type")
    post_type = post_type_element.text if post_type_element is not None else ""

    # check that content is one of the desired post types
    if post_type in target_post_types:
        title = (
            item.find("title").text if item.find("title") is not None else "No Title"
        )

        cleaned_content = ""

        # glossary terms are a special case, as they have multiple definitions so we need to extract them all
        # this means this code is the most "hard coded" and if ever content about glossary terms changes, this code will need to be updated
        if post_type == "glossary-term":
            definitions = []
            for meta in item.findall(
                "wp:postmeta", namespaces={"wp": "http://wordpress.org/export/1.2/"}
            ):
                meta_key = meta.find(f"{wp_namespace}meta_key").text
                meta_value = (
                    meta.find(f"{wp_namespace}meta_value").text
                    if meta.find(f"{wp_namespace}meta_value") is not None
                    else ""
                )

                # Currently, glossary content is stored under "drafting_notes" and "term_definition_" meta keys
                if meta_key == "drafting_notes" or meta_key.startswith(
                    "term_definition_"
                ):
                    # Use beautiful soup to parse HTML and remove tags
                    soup = BeautifulSoup(meta_value, "html.parser")
                    definitions.append(soup.get_text(separator="\n").strip())
            cleaned_content = "\n\n".join(definitions)

        else:
            # for other post types, we can just use the main content field
            content_element = item.find(f"{content_namespace}encoded")
            if content_element is not None and content_element.text:
                # Use beautiful soup to parse HTML and remove tags
                soup = BeautifulSoup(content_element.text, "html.parser")
                cleaned_content = soup.get_text(separator="\n").strip()

        # Saving all files to the output directory
        if cleaned_content.strip():
            filename = f"{title[:50].replace(' ', '_').replace('/', '_')}.txt"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as file:
                file.write(cleaned_content)
                counter += 1

            print(f"Saved: {filepath}")
        else:
            print(f"No content to save for {title}")

print(f"Saved {counter} files")
