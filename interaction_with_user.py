from nltk.stem import WordNetLemmatizer
from torch.utils.data import DataLoader
import ChatBotModel
import numpy as np
import pickle
import torch
import nltk

lemmatizer = WordNetLemmatizer()

with open('data/words.pkl', 'rb') as file:
    words = pickle.load(file)

with open('data/labels.pkl', 'rb') as file:
    labels = pickle.load(file)

with open('data/responses.pkl', 'rb') as file:
    responses = pickle.load(file)

with open('data/patterns.pkl', 'rb') as file:
    patterns = pickle.load(file)

with open('data/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

def user_input_nltk(input_user):
    '''
    Cleaning up user's sentence
    :param input_user: The text given by a user
    :return: Processed text
    '''

    input_user = nltk.word_tokenize(input_user)

    for x in range(len(input_user)):

        input_user[x] = lemmatizer.lemmatize(input_user[x])
        input_user[x] = input_user[x].lower()

    return input_user

bag_of_words = []


def create_bag_of_words(user_input):
    '''
    Creating the bag of words for a processed input
    :param user_input: Processed user input
    :return: Bag of words
    '''
    for x in range(len(user_input)):
        bag_of_words_row = [0] * len(words)
        for y in range(len(words)):

            if user_input[x] == words[y]:
                bag_of_words_row[y] += 1

    bag_of_words.append(bag_of_words_row)

    return bag_of_words

def get_prediction(input_user):

    '''
    Predicting the right tag for response
    :param input_user: Bag of words created from the user's input
    :return: Predicted tag
    '''

    input_user = np.array(input_user)

    input_user = torch.tensor(input_user, dtype=torch.float32)

    user_input = DataLoader(input_user)

    model = ChatBotModel.ChatBotModel(len(words), len(labels))
    model.load_state_dict(torch.load('data/ChatBotModel', weights_only=False))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    with torch.no_grad():
        for idx, (data) in enumerate(user_input):
            data = data.to(device)
            output = model(data)
            max_val = torch.max(output).item()
            best_choice = torch.argmax(output, dim=1)
            if max_val < 0.3:
                best_choice = None
            else:
                best_choice = best_choice.item()

    return best_choice


def answer(input_user):
    '''
    Generating a response for user
    :param input_user: Input
    :return: Answer
    '''
    user_input = user_input_nltk(input_user)
    user_input = create_bag_of_words(user_input)
    if np.all(np.array(user_input)  == 0 ):
        return "Sorry, I don't know how to help you :("
    index = get_prediction(user_input)

    if index is None:
        return "Sorry, I don't know how to help you :("

    label = [key for key in label_encoder.keys() if label_encoder[key] == index]
    label = label[0]
    response = responses[label]
    final_answer = response[0]
    return final_answer



