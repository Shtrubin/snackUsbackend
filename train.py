import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = [] 
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern) 
        all_words.extend(w) 
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag) 
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

num_epochs = 1000
batch_size = 8 
learning_rate = 0.001 
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, 
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

model = NeuralNet(input_size, hidden_size, output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def cross_entropy_loss_manual(outputs, labels, output_size):
    N = labels.shape[0]
    loss = 0
    for i in range(N):
        true_label = labels[i]
        predicted_probs = torch.softmax(outputs[i], dim=-1)
        loss -= torch.log(predicted_probs[true_label])
    return loss / N

def accuracy_manual(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0) * 100
    return accuracy

for epoch in range(num_epochs):
    total_loss = 0
    total_accuracy = 0
    for (words, labels) in train_loader:
        words, labels = words.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(words)
        loss = cross_entropy_loss_manual(outputs, labels, output_size)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_accuracy += accuracy_manual(outputs, labels)

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%')

model.eval()
with torch.no_grad():
    all_predictions = []
    all_labels = []
    for (words, labels) in train_loader:
        words, labels = words.to(device), labels.to(device)
        outputs = model(words)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean() * 100
    print(f'Final Accuracy: {accuracy:.2f}%')

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tags, yticklabels=tags)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print(f'final loss: {loss.item():.4f}')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'training complete. file saved to {FILE}')
