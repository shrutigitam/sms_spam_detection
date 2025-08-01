from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load the dataset
df_spam = pd.read_csv('spam.csv', encoding='latin-1').iloc[:, :2]
df_spam.columns = ['label', 'message']
df_spam['label'] = df_spam['label'].map({'ham': 0, 'spam': 1})

df_header = pd.read_excel('list_header.xlsx')
valid_headers = df_header['Header'].tolist()

# Preprocessing
tfidf_transformer = TfidfTransformer()
count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(df_spam['message'])
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# Train model
model = MultinomialNB()
model.fit(X_tfidf, df_spam['label'])

def is_spam(header, message):
    header_is_spam = header not in valid_headers
    message_vector = count_vect.transform([message])
    message_is_spam = model.predict(message_vector)[0] == 1
    return "spam" if header_is_spam or message_is_spam else "not spam"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    header = data.get('header', '')
    message = data.get('message', '')
    result = is_spam(header, message)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
