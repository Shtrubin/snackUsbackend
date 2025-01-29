import spacy

nlp = spacy.load("en_core_web_sm")

food_list = [
    "biryani", "momo", "pizza", "burger", "pasta", "sushi", "tacos", "ramen", "steak", 
    "sandwich", "ice cream", "cake", "fries", "noodles", "pancakes", "chicken", "fish", "rice", "dumplings", "Espresso", "chatamari","newari","thali","fish","pork"
]

adjective_list = [
    "delicious", "great", "amazing", "excellent", "wonderful", "superb", "awesome", 
    "fabulous", "perfect", "impressive", "terrific", "fantastic", 
    "outstanding", "marvelous", "lovely", "pleasing", "best", "good", 
    "bad", "poor", "worst", "horrible", "terrible", "awful"
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

    for word in text.split():
        if word.lower() in adjective_list:
            if word.lower() not in adjectives:
                adjectives.append(word.lower())

    for food in food_list:
        if food.lower() in text.lower():  
            foods.append(food)

    return places, foods, adjectives
