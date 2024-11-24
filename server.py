from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import random
import json
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from entity_extraction import extract_entities  # Import entity extraction function

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # This is to enable CORS (Cross-Origin Resource Sharing) for frontend

# Load intents and model data
with open('intents.json', 'r') as json_data:
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

@app.route("/chat", methods=["POST"])
def chat():
    message = request.json["message"]
    response = get_response(message)
    return jsonify({"response": response})

def get_response(msg):
    # Call the entity extraction function
    places, foods, adjectives = extract_entities(msg)

    # Print the extracted entities to the console
    print(f"\nEntities detected from message: '{msg}'")
    print("Detected places:", places)
    print("Detected foods:", foods)
    print("Detected adjectives:", adjectives)

    # Proceed with intent recognition
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."

if __name__ == "__main__":
    app.run(port=5000)
