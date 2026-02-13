import spacy

nlp = spacy.load("en_core_web_sm")

def extract_aspects(text):
    doc = nlp(text)
    aspects = []
    for chunk in doc.noun_chunks:
        # Filter out short/useless aspects (this, it, that, etc.)
        if len(chunk.text) > 2 and chunk.text.lower() not in ["this", "it", "that", "what", "which", "who", "one"]:
            aspects.append(chunk.text)
    return list(set(aspects))
