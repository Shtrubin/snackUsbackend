import re
import torch
from nltk_utils import bag_of_words, tokenize, stem 
from entity_extraction import extract_entities
from model import NeuralNet
import json
import random
import mysql.connector

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

def preprocess_message(msg):
    tokenized_msg = tokenize(msg)
    processed_msg = " ".join([stem(word) for word in tokenized_msg])
    return processed_msg

def extract_place_name(msg):
    match = re.search(r'\bin\s+([a-zA-Z0-9\s]+)', msg)
    if match:
        return match.group(1).strip() 
    return None

def get_response(msg, db):
    msg = preprocess_message(msg) 

    intent_response = process_intent_with_model(msg, db)
    if intent_response:
        return intent_response
    
    return "I do not understand. Could you clarify your request?"

def process_intent_with_model(msg, db=None):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    places, foods, adjectives = extract_entities(msg.title())

    if prob.item() > 0.6:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "find_restaurant":

                    if len(places) > 0:
                        return fetch_restaurants_by_location(places[0], db)
                    else:
                        return random.choice(intent['responses'])

                elif tag == "find_by_category":
                    category = extract_category(msg)
                    if category:
                        return fetch_restaurants_by_category(category, db)
                    else:
                        return random.choice(intent['responses'])

                elif tag == "find_by_special_item":
                    if len(foods) > 0:
                        return fetch_restaurants_by_special_item(foods[0], db)
                    else:
                        return random.choice(intent['responses'])

                elif tag == "find_by_rating":
                    rating = extract_rating(msg)
                    if rating:
                        return fetch_restaurants_rating(rating, db)
                    else:
                        return random.choice(intent['responses'])

                elif tag == "greeting":
                    return random.choice(intent['responses'])

                elif tag == "thank_you":
                    return random.choice(intent['responses'])

                elif tag == "goodbye":
                    return random.choice(intent['responses'])

    else:
        return "I couldn't understand..."

def fetch_restaurants_by_category(category, db=None):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, location, rating FROM restaurants WHERE category = %s LIMIT 3", (category,))
    result = cursor.fetchall()

    if result:
        response = f"Here are some great restaurants in the {category} category:\n"
        for row in result:
            response += f"{row[0]} - {row[1]} - Rating: {row[2]}\n"
        return response
    return "Sorry, I couldn't find any restaurants in that category."

def fetch_restaurants_by_special_item(special_item, db=None):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, special_item, description FROM restaurants WHERE special_item LIKE %s LIMIT 3", ('%' + special_item + '%',))
    result = cursor.fetchall()

    if result:
        response = f"Here are some restaurants that serve {special_item}:\n"
        for row in result:
            response += f"{row[0]} - Special Item: {row[1]}\nDescription: {row[2]}\n\n"
        return response
    return f"Sorry, I couldn't find any restaurants serving {special_item}."

def fetch_restaurants_rating(rating, db=None):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, rating FROM restaurants WHERE rating >= %s LIMIT 3", (rating,))
    result = cursor.fetchall()

    if result:
        response = f"Here are some restaurants with a rating above {rating}:\n"
        for row in result:
            response += f"{row[0]} - Rating: {row[1]}\n"
        return response
    return f"Sorry, I couldn't find any restaurants with a rating above {rating}."

def fetch_restaurants_by_location(location, db=None):
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name, rating, description FROM restaurants WHERE location = %s OR sub_location LIKE %s LIMIT 3", (location, location))
    result = cursor.fetchall()

    if result:
        response = f"Here are some great restaurants in {location}:\n"
        for row in result:
            response += f"{row[0]} - Rating: {row[1]}\nDescription: {row[2]}\n\n"
        return response
    else:
        return f"Sorry, I couldn't find any restaurants in {location}."

def extract_category(msg):
    categories = {
        'mid': 'mid-range', 'mid range': 'mid-range', 'mid-range': 'mid-range',
        'high': 'high-end', 'high end': 'high-end', 'high-end': 'high-end',
        'local': 'local'
    }
    
    for category, standardized_category in categories.items():
        if category in msg.lower():
            return standardized_category
    
    return None

def extract_rating(msg):
    match = re.search(r"\b([1-9]|10)\b", msg)
    if match:
        return int(match.group(0))
    return None
