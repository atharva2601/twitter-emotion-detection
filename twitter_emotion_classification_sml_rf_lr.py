import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load datasets
training_data = pd.read_csv('Dataset/training.csv')
validation_data = pd.read_csv('Dataset/validation.csv')
test_data = pd.read_csv('Dataset/test.csv')

"""Number of Class counts"""

plt.figure(figsize=(10, 6))
sns.countplot(data=training_data, x='label')
plt.title('Distribution of Classes')
plt.show()

text = ' '.join(training_data['text'])
wordcloud = WordCloud(background_color='white', max_words=100, width=800, height=400).generate(text)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Define the order and names of your classes
class_names = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}

# Map numeric labels to emotion names
training_data['emotion'] = training_data['label'].map(class_names)

emotion_distribution = training_data['emotion'].value_counts()

# Plotting the pie chart for class distribution
plt.figure(figsize=(8, 8))
patches, texts, autotexts = plt.pie(emotion_distribution, labels=emotion_distribution.index, autopct='%1.1f%%', startangle=140)

# Improve readability for the percentage labels
for text in autotexts:
    text.set_color('white')

# Add a legend
plt.legend(patches, emotion_distribution.index, loc='best', title='Emotions')
plt.title('Emotion Distribution')
plt.show()

"""## **SVM**"""

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('svm', SVC(kernel='linear', C=1))
])

# Combine training and validation data
X_train = pd.concat([training_data['text'], validation_data['text']])
y_train = pd.concat([training_data['label'], validation_data['label']])
X_test = test_data['text']
y_test = test_data['label']

# Train the model
pipeline.fit(X_train, y_train)

"""**Accuracy**"""

predictions = pipeline.predict(X_test)
print(classification_report(y_test, predictions))
print("SVM Accuracy:", accuracy_score(y_test, predictions))

# Combine train and validation data for training after hyperparameter tuning
full_train_data = pd.concat([training_data, validation_data])

# Vectorization
vectorizer = TfidfVectorizer()
X_full_train = vectorizer.fit_transform(full_train_data['text'])
X_test = vectorizer.transform(test_data['text'])

# Labels
y_full_train = full_train_data['label']
y_test = test_data['label']

"""## **Logistic Regression**"""

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_full_train, y_full_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

"""**Accuracy**"""

print(classification_report(y_test, lr_predictions))

print(f"Logistic Regression Accuracy: {lr_accuracy}")

"""## **Random Forest**"""

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_full_train, y_full_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

"""**Accuracy**"""

print(classification_report(y_test, rf_predictions))

print(f"Random Forest Accuracy: {rf_accuracy}")