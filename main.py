import tensorflow as tf
import os

# define paths to data files
masterdict_dir = 'MasterDictionary'
pos_file = os.path.join(masterdict_dir, 'positive-words.txt')
neg_file = os.path.join(masterdict_dir, 'negative-words.txt')

# read in the positive and negative word lists
with open(pos_file, 'r', encoding='latin-1') as f:
    positive_words = f.read().splitlines()

with open(neg_file, 'r', encoding='latin-1') as f:
    negative_words = f.read().splitlines()

# create a TensorFlow tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()

def tokenize_text(text):
    # tokenize the text
    tokenizer.fit_on_texts([text])
    tokens = tokenizer.texts_to_sequences([text])[0]
    
    # return the tokens
    return tokens

# define function to calculate sentiment scores
def calculate_sentiment(text):
    # tokenize the text
    tokens = tokenize_text(text)
    
    # calculate positive and negative scores
    pos_score = sum([1 for token in tokens if tokenizer.index_word[token] in positive_words])
    neg_score = sum([1 for token in tokens if tokenizer.index_word[token] in negative_words])
    
    # calculate polarity score
    polarity_score = (pos_score - neg_score) / ((pos_score + neg_score) + 0.000001)
    
    # calculate subjectivity score
    subjectivity_score = (pos_score + neg_score) / (len(tokens) + 0.000001)
    
    # return scores
    return pos_score, neg_score, polarity_score, subjectivity_score

# define function to determine overall sentiment
def determine_overall_sentiment(polarity_score):
    if polarity_score > 0.3:
        return "Positive"
    elif polarity_score < -0.3:
        return "Negative"
    else:
        return "Neutral"

# define function to calculate accuracy
def calculate_accuracy(test_texts, test_labels):
    correct_predictions = 0
    total_predictions = 0
    
    for text, label in zip(test_texts, test_labels):
        _, _, polarity_score, _ = calculate_sentiment(text)
        predicted_label = determine_overall_sentiment(polarity_score)
        
        if predicted_label == label:
            correct_predictions += 1
        total_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy

test_texts = [
    "This movie is an absolute masterpiece! The acting, the cinematography, and the storyline are all top-notch.",
    "I'm so disappointed with this product. It's poorly made and doesn't work as advertised.",
    "The weather forecast for today is sunny with a high of 75 degrees.",
    "I had an amazing time at the concert last night! The band's energy was incredible.",
    "This restaurant is terrible. The food was bland, and the service was slow.",
    "Congratulations to the new graduates! Your hard work has paid off.",
    "I'm frustrated with the traffic situation in this city. It's a nightmare during rush hour.",
    "Have you tried the new coffee shop? Their lattes are heavenly!",
    "I'm sorry to hear about your loss. My deepest condolences.",
    "The new software update has been a disaster. It's full of bugs and crashes constantly.",
    "The beach was beautiful today. The water was crystal clear, and the sand was so soft.",
    "I'm worried about the current political situation. It's causing a lot of division and tension.",
    "This book is a real page-turner! I couldn't put it down.",
    "The customer service at this company is appalling. They're rude and unhelpful.",
    "I'm excited for the upcoming holiday season! It's my favorite time of the year.",
    "The new tax laws are unfair and will hurt many hard-working families.",
    "I had a wonderful time at the art gallery. The exhibitions were thought-provoking and inspiring.",
    "This laptop is a piece of junk. It's slow and constantly freezes.",
    "The park was so peaceful and serene. It's the perfect place to relax and unwind.",
    "I'm outraged by the recent police brutality incident. It's unacceptable, and changes need to be made.",
]

test_labels = [
    "Positive", "Negative", "Neutral", "Positive", "Negative", "Positive", "Negative", "Positive",
    "Neutral", "Negative", "Positive", "Negative", "Positive", "Negative", "Positive", "Negative",
    "Positive", "Negative", "Positive", "Negative",
]

# calculate accuracy
accuracy = calculate_accuracy(test_texts, test_labels)
print(f"Accuracy: {accuracy * 100}%")

# prompt the user for input text
input_text = input("Enter the text to analyze: ")

# calculate sentiment scores
pos_score, neg_score, polarity_score, subjectivity_score = calculate_sentiment(input_text)
# determine overall sentiment and emoji
sentiment_label = determine_overall_sentiment(polarity_score)

# print the scores
# print the sentiment results with emoji
print(f"Overall Sentiment: {sentiment_label}")
print(f"Positive Score: {pos_score}")
print(f"Negative Score: {neg_score}")
print(f"Polarity Score: {polarity_score}")
print(f"Subjectivity Score: {subjectivity_score}")