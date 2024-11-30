import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Predefined list of common foods (can be extended)
food_list = [
    "biryani", "momo", "pizza", "burger", "pasta", "sushi", "tacos", "ramen", "steak", 
    "sandwich", "ice cream", "cake", "fries", "noodles", "pancakes", "chicken", "fish", "rice", "dumplings", "Espresso"
]

# Function to extract entities
def extract_entities(text):     

    doc = nlp(text)
    places = []
    foods = []
    adjectives = []
    
    # Extract places using spaCy's NER (GPE)
    for ent in doc.ents:
        if ent.label_ == 'GPE':  # Geopolitical Entity (Place)
            places.append(ent.text)

    # Extract adjectives (spaCy POS tagging)
    for token in doc:
        if token.pos_ == 'ADJ':  # Adjective POS tag
            adjectives.append(token.text)

    # Match food items based on predefined food list
    for food in food_list:
        if food.lower() in text.lower():  # Case insensitive match
            foods.append(food)

    return places, foods, adjectives
