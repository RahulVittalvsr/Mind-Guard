import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

# --- CONFIGURATION ---
print("Initializing MindGuard...")
nltk.download('stopwords', quiet=True)

# 1. LOAD DATASET
try:
    data = pd.read_csv("mental_stress_data.csv")
except FileNotFoundError:
    print("ERROR: 'mental_stress_data.csv' not found!")
    exit()

# 2. TEXT PREPROCESSING
stop_words = set(stopwords.words('english'))
# Remove negatives from stopwords so "not happy" isn't read as "happy"
for w in ['not', 'no', 'nor', 'never', 'dont', 'cant']:
    stop_words.discard(w)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data["clean_text"] = data["text"].apply(clean_text)

# 3. TRAIN MODEL (With Bigrams!)
# ngram_range=(1,2) lets it understand "not happy" as a phrase, not just words.
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(data["clean_text"]).toarray()
y = data["label"]

model = MultinomialNB()
model.fit(X, y)

# 4. PREDICTION FUNCTION (With Safety Net)
def predict_stress(user_text):
    text_lower = user_text.lower()
    
    # --- SAFETY NET: Force correct answers for your specific short keywords ---
    # This guarantees the "Slap Protection" logic you asked for.
    
    stress_triggers = ['stress', 'bad', 'worse', 'not good', 'pain', 'sad', 'die', 'kill', 'hopeless']
    happy_triggers = ['happy', 'good', 'wow', 'great', 'calm', 'relax', 'joy', 'best', 'fine']
    unsure_triggers = ['dont know', 'unsure', 'maybe', 'idk', 'dunno']

    # Check for "Unsure" first
    if any(w in text_lower for w in unsure_triggers):
        return "unsure"

    # Check for specific "Stress" words
    if any(w in text_lower for w in stress_triggers):
        # Double check it's not "not stress"
        if "no stress" not in text_lower and "not stressed" not in text_lower:
            return "stress"

    # Check for specific "Happy" words
    if any(w in text_lower for w in happy_triggers):
        # Double check it's not "not happy"
        if "not happy" not in text_lower and "not good" not in text_lower:
            return "no_stress"

    # --- AI MODEL BACKUP (For complex sentences) ---
    cleaned = clean_text(user_text)
    # If cleaning removed everything (e.g. user typed "the the"), return unsure
    if not cleaned:
        return "unsure"
        
    vector = vectorizer.transform([cleaned]).toarray()
    proba = model.predict_proba(vector)[0]
    prediction = model.predict(vector)[0]

    # If the AI is confused (confidence low), return unsure
    if max(proba) < 0.55:
        return "unsure"
        
    return prediction

# 5. USER INTERACTION LOOP
print("\n" + "="*40)
print("   MindGuard: Mental Stress Detector")
print("="*40)

while True:
    user_input = input("\nHow are you feeling? (type 'exit' to quit): ")

    if user_input.lower() == "exit":
        print("Take care! 💙")
        break

    result = predict_stress(user_input)

    if result == "stress":
        print("⚠️  Result: High Mental Stress Detected")
        print("   -> Suggestion: Relax, take a deep breath.")
    elif result == "no_stress":
        print("😊 Result: No Significant Stress Detected")
        print("   -> Great! Keep up the positive vibes.")
    else:
        print("🤔 Result: Unsure")
        print("   -> I didn't quite get that. Can you say it differently?")