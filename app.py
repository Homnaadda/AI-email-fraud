# ==========================================================
# 1. One-time installs (uncomment if libraries missing)
# ==========================================================
# !pip install -q scikit-learn==1.4.2 imbalanced-learn==0.12.3 joblib==1.4.2

# ==========================================================
# 2. Imports
# ==========================================================
import pandas as pd
import re, string, nltk, joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline   # keeps SMOTE inside CV
from imblearn.over_sampling import SMOTE

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOP = set(stopwords.words('english'))

# ==========================================================
# 3. Load KD-10k
# ==========================================================
df = pd.read_csv('dataset/phishing_legit_dataset_KD_10000.csv')
print(df['label'].value_counts())          # quick sanity check

# ==========================================================
# 4. Minimal text cleaner
# ==========================================================
def clean(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' URL ', text)          # normalise links
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', ' NUM ', text)
    tokens = [t for t in text.split() if t not in STOP and len(t) > 2]
    return ' '.join(tokens)

df['text_clean'] = df['text'].apply(clean)

# ==========================================================
# 5. Stratified split (keeps label ratio)
# ==========================================================
X, y = df['text_clean'], df['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================================
# 6. Build pipeline: TF-IDF → Chi² → SMOTE → LogReg
#    (SMOTE lives *inside* CV ⇒ no test leakage)
# ==========================================================
pipe = ImbPipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=20_000,
                              ngram_range=(1,2),
                              min_df=2,
                              stop_words='english')),
    ('chi',   SelectKBest(chi2, k=12_000)),   # aggressive cut
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('clf',   LogisticRegression(max_iter=1000,
                                 class_weight='balanced',
                                 solver='liblinear'))
])

# ==========================================================
# 7. Tiny but smart grid (30 sec on CPU)
# ==========================================================
param_grid = {
    'clf__C': [0.5, 1, 2],
    'clf__penalty': ['l2'],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe,
                    param_grid,
                    cv=cv,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=0)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("CV F1:", f"{grid.best_score_:.3f}")

# ==========================================================
# 8. Evaluate on untouched test set
# ==========================================================
pred = grid.best_estimator_.predict(X_test)
print(classification_report(y_test, pred, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

# ==========================================================
# 9. Persist pipeline
# ==========================================================
joblib.dump(grid.best_estimator_, 'phishing_pipeline.joblib')
print("Model saved → phishing_pipeline.joblib")

# ==========================================================
# 10. One-liner helper for live inference
# ==========================================================
def predict_phishing(text, model_path='phishing_pipeline.joblib'):
    clf = joblib.load(model_path)
    return clf.predict_proba([clean(text)])[0,1]   # phishing probability

# Quick demo
sample = "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim now."
print(f"Phishing score: {predict_phishing(sample):.2%}")