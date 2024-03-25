import streamlit as st
import tensorflow as tf
import os

# Define paths to data files
masterdict_dir = 'MasterDictionary'
pos_file = os.path.join(masterdict_dir, 'positive-words.txt')
neg_file = os.path.join(masterdict_dir, 'negative-words.txt')

# Read in the positive and negative word lists
with open(pos_file, 'r', encoding='latin-1') as f:
    positive_words = f.read().splitlines()

with open(neg_file, 'r', encoding='latin-1') as f:
    negative_words = f.read().splitlines()

# Create a TensorFlow tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()

def tokenize_text(text):
    # Tokenize the text
    tokenizer.fit_on_texts([text])
    tokens = tokenizer.texts_to_sequences([text])[0]
    
    # Return the tokens
    return tokens

# Define function to calculate sentiment scores
def calculate_sentiment(text):
    # Tokenize the text
    tokens = tokenize_text(text)
    
    # Calculate positive and negative scores
    pos_score = sum([1 for token in tokens if tokenizer.index_word[token] in positive_words])
    neg_score = sum([1 for token in tokens if tokenizer.index_word[token] in negative_words])
    
    # Calculate polarity score
    polarity_score = (pos_score - neg_score) / ((pos_score + neg_score) + 0.000001)
    
    # Calculate subjectivity score
    subjectivity_score = (pos_score + neg_score) / (len(tokens) + 0.000001)
    
    # Return scores
    return pos_score, neg_score, polarity_score, subjectivity_score

# Define function to determine overall sentiment
def determine_overall_sentiment(polarity_score):
    if polarity_score > 0.3:
        return "Positive"
    elif polarity_score < -0.3:
        return "Negative"
    else:
        return "Neutral"

# Define Streamlit app
def main():
    st.title("Sentiment Analysis with Emoji")

    # Input text box
    input_text = st.text_area("Enter the text to analyze:", "")

    # Analyze button
    if st.button("Analyze"):
        if input_text.strip() == "":
            st.error("Please enter some text.")
        else:
            # Calculate sentiment scores
            pos_score, neg_score, polarity_score, _ = calculate_sentiment(input_text)
            # Determine overall sentiment
            sentiment_label = determine_overall_sentiment(polarity_score)

            # Display scores
            st.write(f"Overall Sentiment: {sentiment_label}")
            st.write(f"Positive Score: {pos_score}")
            st.write(f"Negative Score: {neg_score}")
            st.write(f"Polarity Score: {polarity_score}")

            # Display emoji based on sentiment
            if sentiment_label == "Positive":
                st.write("😃")
            elif sentiment_label == "Negative":
                st.write("😔")
            else:
                st.write("😐")

if __name__ == "__main__":
    main()
