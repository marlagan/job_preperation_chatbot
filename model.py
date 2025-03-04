from torch.utils.data import DataLoader, Dataset
from nltk.stem import WordNetLemmatizer
import torch.optim as optim
from torch import nn
import ChatBotModel
import numpy as np
import pickle
import string
import torch
import json
import nltk

words = []
patterns = []
labels = []
responses = {}
label_encoder = {}

nltk.download('stopwords')

with open("data/data.json", "rb") as file:
    json_file = json.load(file)

lemmatizer = WordNetLemmatizer()
counter = 1

for intent in json_file['intents']:
    for pattern in intent['patterns']:
        p = []
        tokens = nltk.word_tokenize(pattern)
        for x in tokens:
            if x not in string.punctuation:
                if x not in words:
                    words.append(x)
                p.append(x)
        p.append((intent['tag']))
        patterns.append(p)

        if intent['tag'] not in label_encoder.keys():
            label_encoder[intent['tag']] = counter
            counter += 1

        labels.append(label_encoder[intent['tag']])

    for response in intent['responses']:
        if intent['tag'] not in responses:
            responses[intent['tag']] = []

        responses[intent['tag']].append(response)

for x in range(len(words)):
    words[x] = lemmatizer.lemmatize(words[x])
    words[x] = words[x].lower()

bag_of_words = []

for pattern in patterns:

    bag_of_words_row = [0] * len(words)
    for x in range(len(pattern)-1):

        pattern[x] = lemmatizer.lemmatize(pattern[x])
        pattern[x] = pattern[x].lower()

        for y in range(len(words)):

            if pattern[x] == words[y]:
                bag_of_words_row[y] += 1

    bag_of_words.append(bag_of_words_row)


with open('data/labels.pkl', 'wb') as file:
    pickle.dump(labels, file)

with open('data/words.pkl', 'wb') as file:
    pickle.dump(words, file)

with open('data/responses.pkl', 'wb') as file:
    pickle.dump(responses, file)

with open('data/patterns.pkl', 'wb') as file:
    pickle.dump(patterns, file)

with open('data/label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)


bag_of_words = np.array(bag_of_words)

labels = np.array(labels)

train_x = bag_of_words
train_y = labels

class ChatDataset(Dataset):

    def __init__(self, train_x, train_y):

        self.train_x = torch.tensor(train_x, dtype=torch.float32)
        self.train_y = torch.tensor(train_y, dtype=torch.long)
        self.len = len(self.train_x)

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]

    def __len__(self):
        return self.len

batch_size = 1
train_dataset = ChatDataset(train_x, train_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChatBotModel.ChatBotModel(len(train_x[0]), len(train_y)).to(device)
loss_en = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    '''
    Method responsible for training the model
    :param epoch: the current epoch
    :return:
    '''
    accuracy = 0
    model.train()
    for idx,(data,label) in enumerate(train_loader):

        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_en(output, label)
        loss.backward()
        optimizer.step()
        accuracy += (output.argmax(dim=1) == label).float().sum()
    accuracy = 100 * accuracy / len(train_loader)

    print(f'{epoch} - accuracy = {accuracy}')


for epoch in range(90):
    train(epoch)

torch.save(model.state_dict(), 'data/ChatBotModel')

