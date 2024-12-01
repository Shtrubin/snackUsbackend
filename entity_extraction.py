import spacy

nlp = spacy.load("en_core_web_sm")

food_list = [
    "biryani", "momo", "pizza", "burger", "pasta", "sushi", "tacos", "ramen", "steak", 
    "sandwich", "ice cream", "cake", "fries", "noodles", "pancakes", "chicken", "fish", "rice", "dumplings", "Espresso"
]

def extract_entities(text):     

    doc = nlp(text)
    places = []
    foods = []
    adjectives = []
    
    for ent in doc.ents:
        if ent.label_ == 'GPE':  
            places.append(ent.text)

    for token in doc:
        if token.pos_ == 'ADJ': 
            adjectives.append(token.text)

    for food in food_list:
        if food.lower() in text.lower():  
            foods.append(food)

    return places, foods, adjectives
