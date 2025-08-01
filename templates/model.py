import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import os
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ensure required packages are installed
def install_missing_packages():
    try:
        import matplotlib
        import seaborn
        import wordcloud
        import sklearn
        import openpyxl
    except ImportError:
        os.system(f"{sys.executable} -m pip install matplotlib seaborn wordcloud scikit-learn pandas openpyxl")
        print("Missing packages installed. Please restart the script.")
        sys.exit()

install_missing_packages()

# Load Datasets
df_spam = pd.read_csv('spam.csv', encoding='latin-1')
df_header = pd.read_excel('list_header.xlsx', engine='openpyxl')

# Process Spam Dataset
df_spam = df_spam.iloc[:, :2]  # Retain only the first two columns
df_spam.columns = ['label', 'message']
df_spam['label'] = df_spam['label'].map({'ham': 0, 'spam': 1})

# Process Header Dataset
valid_headers = df_header['Header'].tolist()

# Preprocessing: Text to feature vectors
X = df_spam['message']
y = df_spam['label']

count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(X)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model and vectorizer
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('count_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(count_vect, vectorizer_file)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# WordCloud Visualization
spam_words = ' '.join(df_spam[df_spam['label'] == 1]['message'])
ham_words = ' '.join(df_spam[df_spam['label'] == 0]['message'])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(WordCloud(background_color='white', max_words=200).generate(spam_words))
plt.axis('off')
plt.title('Spam Messages WordCloud')

plt.subplot(1,2,2)
plt.imshow(WordCloud(background_color='white', max_words=200).generate(ham_words))
plt.axis('off')
plt.title('Ham Messages WordCloud')
plt.show()

# Function to check if an email is spam
def is_spam(header, message, valid_headers, spam_model, vectorizer):
    header_is_spam = header not in valid_headers
    message_vector = vectorizer.transform([message])
    message_is_spam = spam_model.predict(message_vector)[0] == 1
    return "spam" if header_is_spam or message_is_spam else "not spam"

# Interactive spam detection
while True:
    header = input("Enter the email header (or type 'stop' to quit): ")
    if header.lower() == "stop":
        print("Exiting the program.")
        break
    message = input("Enter the email message: ")
    if message.lower() == "stop":
        print("Exiting the program.")
        break
    result = is_spam(header, message, valid_headers, model, count_vect)
    print(f"The result is: {result}")
