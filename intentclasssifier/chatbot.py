import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import json
import re

# load intents data
from textblob import TextBlob

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

all_genres = set()
for genres_str in movies['genres'].str.split('|'):
    all_genres.update([genre.lower() for genre in genres_str])

# Convert set to list and sort
genres_list = list(all_genres)
# print(genres_list)
movies['genres'] = movies['genres'].str.replace('|', ' ')
vectorizer = TfidfVectorizer(stop_words='english')
genre_matrix = vectorizer.fit_transform(movies['genres'])


def load_data():
    with open('intents.json', 'r') as f:
        intents = json.load(f)

    # preprocess the data
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in intents:
        for example in intent['examples']:
            # tokenize each example
            tokens = example.lower().split()
            words.extend(tokens)
            docs_x.append(tokens)
            docs_y.append(intent['intent'])
        # add unique labels
        if intent['intent'] not in labels:
            labels.append(intent['intent'])

    # create a vocabulary set
    words = sorted(list(set(words)))

    # create bag of words
    training_data = []
    for i, doc in enumerate(docs_x):
        bag = []
        for word in words:
            bag.append(1) if word in doc else bag.append(0)
        output_row = [0] * len(labels)
        output_row[labels.index(docs_y[i])] = 1
        training_data.append([bag, output_row])

    # shuffle the training data
    random.shuffle(training_data)

    # split the training and testing data
    train_x = np.array([data[0] for data in training_data])
    train_y = np.array([data[1] for data in training_data])
    return words, labels, train_x, train_y


# build the neural network model
def build_model(train_x, train_y):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model


# function to predict intent
def predict_intent(text, words, labels, model):
    # tokenize the text
    tokens = text.lower().split()
    # create a bag of words
    bag = [1 if word in tokens else 0 for word in words]
    # predict the intent using the trained model
    prediction = model.predict(np.array([bag]), verbose=0)[0]
    # get the predicted label
    label = labels[np.argmax(prediction)]
    # get the confidence score
    score = round(np.max(prediction), 2)
    # print(label, 'Label')
    if label == 'recommendation':
        responses = ['BOT: What movies do you like?']
        response = random.choice(responses)
        # if response == responses[0]:
        #     return response
        # else:
        print(response)
        movie = input("You: ")
        #print(movie)
        return recommend_similar_movies(movie)
    elif label == 'genre_preference':
        # print('here')
        sentiment = predict_sentiment(text)
        if sentiment == 'positive' or sentiment == 'neutral':
            words = text.split()

            for word in words:
                if word.lower() in genres_list:
                    # print('here 2')
                    return recommend_movies(word)
        else:
            return "BOT: I'm sorry to hear that you don't enjoy those genres. What genres do you enjoy?"
    elif label == 'greeting':
        return 'BOT: Hello! How can I help you today?'
    elif label == 'goodbye':
        return 'BOT: Happy to help!'
    elif label == 'top_rated':
        return recommend_top_rated_movies(5)
    else:
        return 'BOT: I did not get it. Please ask another question'


def predict_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"


def recommend_movies(intent_label):
    # Get the indices of movies that match the intent label
    # print(intent_label)
    indices = []
    for i in range(len(movies)):
        if intent_label.lower() in movies.iloc[i]['genres'].lower():
            indices.append(i)

    # Calculate the cosine similarities between the movies and the genre matrix
    # print(genre_matrix)
    similarities = cosine_similarity(genre_matrix[indices], genre_matrix)

    # Get the top 10 most similar movies
    top_indices = similarities.mean(axis=0).argsort()[::-1][:5]
    recommended_movies = movies.iloc[top_indices]
    # print('hhhhh' ,type(recommended_movies))
    # Return the recommended movies
    recommended_movies_list = list(recommended_movies['title'].values)
    return 'BOT: These are my recommendations for ' + intent_label + ' movies \n' + \
           ', '.join(recommended_movies_list)


def recommend_similar_movies(movie_title):
    # convert the movie titles to lower case
    movies['title_mod'] = movies['title'].apply(lambda x: re.sub(r'\(\d{4}\)', '', x).strip()).str.lower()
    if movie_title.lower() not in movies['title_mod'].unique():
        # print(f"Movie '{movie_title}' not found in the dataset.")
        return 'BOT: no such movie found in database. suggest another one'

    movie_index = movies[movies['title_mod'] == movie_title.lower()].index.values[0]

    # calculate cosine similarity between the given movie and all other movies
    similarities = cosine_similarity(genre_matrix[movie_index], genre_matrix)

    # sort the similarities in descending order and get the top 10

    similar_indices = similarities.argsort()[0][-6:-1][::-1]

    # get the titles of the recommended movies
    recommended_movies = movies.iloc[similar_indices]['title']
    # print(recommended_movies)
    recommended_movies_list = recommended_movies.tolist()

    return 'BOT: If you enjoyed  ' + movie_title + ' you would enjoy these movies: \n' + \
           '\n '.join(recommended_movies_list)


def recommend_top_rated_movies(n=5):

    # Calculate the mean rating for each movie
    movie_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()

    # Join the movie data with the mean rating data
    movie_data = pd.merge(movies, movie_ratings, on='movieId')

    # Sort the movies by rating in descending order and return the top n movies
    top_movies = movie_data.sort_values('rating', ascending=False).head(n)

    tm = top_movies[['title', 'rating']]
    print(tm)
    movies_list = tm.values.tolist()
    print(movies_list)
    movies_str = list(map(''.join, movies_list))


    # Return the title and rating of the top movies
    return 'Top 5 high rated movies are ' + ', '.join(movies_str)


def main():
    words, labels, train_x, train_y = load_data()
    model = build_model(train_x, train_y)
    model.fit(train_x, train_y, epochs=500, batch_size=8, verbose=0)
    model.save('intent_classifier.h5')
    print('BOT: Hello! I am your chatbot and I am ready to help you pick a movie!')
    while True:
        text = input("You: ")
        if text.lower() == "quit":
            break

        print(predict_intent(text, words, labels, model))
        # print(f"Bot: {label} ({score})")


main()

# Precision = Number of recommended movies that are relevant / Total number of recommended movies
# Recall = Number of recommended movies that are relevant / Total number of relevant movies
