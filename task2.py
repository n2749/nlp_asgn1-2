import spacy
from spacy import displacy

from get_paragraph import get_paragraph_from_war_and_peace


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

### Named Entity Recognition (NER) using spaCy
def ner_spacy(text: str):
    """Perform Named Entity Recognition using spaCy."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def visualize_ner(text):
    """Perform Named Entity Recognition and visualize entities using displacy."""
    doc = nlp(text)
    displacy.serve(doc, style="ent", host="0.0.0.0", port=5000)


### Main Execution
def main():
    text = get_paragraph_from_war_and_peace()
    print(f"\nSample Paragraph:\n\"\"\"\n{text}\n\"\"\"\n")

    visualize_ner(text)

if __name__ == "__main__":
    main()

