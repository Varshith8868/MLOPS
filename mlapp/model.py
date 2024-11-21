import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files

# Print message indicating the start of dataset loading
print("Loading the dataset...")

# Load dataset
data = load_files('aclImdb/train', categories=['pos', 'neg'], encoding='utf-8')
X, y = data.data, data.target

# Print message after dataset is loaded
print("Dataset loaded. Preprocessing text data...")

# Text preprocessing: Convert text data into numerical format using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model and vectorizer to a file
with open('mlapp/sentiment_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

# Print confirmation message when training is completed and saved
print("Model training completed and saved.")
