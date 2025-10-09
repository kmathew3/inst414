import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Set path to dataset
base_path = r'C:/Users/krist/Downloads/inst414/archive/train'

data = []

# Step 2: Read each text file and label it
for label in ['0', '1']:  # 0 = human, 1 = bot
    folder = os.path.join(base_path, label)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                data.append({'text': text, 'label': int(label)})
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")

# Step 3: Create DataFrame
df = pd.DataFrame(data)
print(f"\nLoaded {len(df)} samples.")
print(df['label'].value_counts())

# Step 4: Preprocessing (very basic cleanup)
df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)  # Remove links
df['text'] = df['text'].str.replace(r'@\w+', '', regex=True)     # Remove mentions
df['text'] = df['text'].str.replace(r'#\w+', '', regex=True)     # Remove hashtags
df['text'] = df['text'].str.lower()                              # Lowercase

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Step 6: Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 7: Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 8: Evaluate the model
y_pred = model.predict(X_test_vec)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))