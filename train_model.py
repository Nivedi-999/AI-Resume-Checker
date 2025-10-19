from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load your dataset (see below for sources)
df = pd.read_csv("resume_dataset.csv")
df = df.rename(columns={'Resume': 'resume_text', 'Category': 'label'})


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['resume_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model accuracy:", model.score(X_test, y_test))

joblib.dump(vectorizer, 'model/tfidf_vectorizer.joblib')
joblib.dump(model, 'model/resume_quality_model.joblib')
