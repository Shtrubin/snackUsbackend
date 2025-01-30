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
locations = ['pokhara', 'patan', 'dharan', 'bhaktapur', 'kathmandu', 'chitwan', 'butwal', 'lalitpur', 'banepa']

def extract_entities(text):     
    doc = nlp(text)
    places = []
    foods = []
    adjectives = []

    print("the entity are", text)
    for location in locations:
        if location in text.lower() and location.capitalize() not in places:
            places.append(location.capitalize())

    for ent in doc.ents:
        if ent.label_ == 'GPE' and ent.text not in places:  
            places.append(ent.text)

    for token in doc:
        if token.pos_ == 'ADJ' and token.text not in adjectives:
            adjectives.append(token.text)

    for word in text.split():
        if word.lower() in adjective_list and word.lower() not in adjectives:
            adjectives.append(word.lower())

    for food in food_list:
        if food.lower() in text.lower() and food.lower() not in foods:  
            foods.append(food)
    print('the enity areeee',places, foods, adjectives )

    return places, foods, adjectives
