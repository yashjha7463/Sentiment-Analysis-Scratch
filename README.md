# Sentiment Analysis with Streamlit

This is a simple Streamlit web application for sentiment analysis. Given a piece of text input, the app calculates sentiment scores and determines whether the sentiment is positive, negative, or neutral. It also displays an emoji corresponding to the sentiment.

## Formula Used

The sentiment analysis is performed using the following formula:
Polarity Score = (Positive Score - Negative Score) / (Positive Score + Negative Score + 0.000001)

Where:
- `Positive Score` is the count of positive words in the text.
- `Negative Score` is the count of negative words in the text.

The `Polarity Score` represents the overall sentiment of the text, ranging from -1 (extremely negative) to 1 (extremely positive).

## Libraries Used

The project utilizes the following Python libraries:

- TensorFlow: For tokenization and text processing.
- Streamlit: For creating the web application interface.
- Pandas: For data manipulation and analysis.
- Altair: For data visualization.
- GitPython: For interacting with Git repositories.
- Others: Various utility libraries for string manipulation, file handling, and data processing.

For the complete list of dependencies, refer to the `requirements.txt` file in the project repository.
