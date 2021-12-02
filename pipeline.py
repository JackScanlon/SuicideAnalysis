from texthero import preprocessing as pp
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import make_scorer, recall_score, accuracy_score, precision_score

# File directory for training input
INPUT_DIRECTORY = './input/'

# Preprocessing pipeline for cleaning text
PREPROCESSING_PIPELINE = [
  pp.remove_html_tags,
  pp.remove_urls,
  pp.lowercase,
  pp.remove_digits,
  pp.remove_punctuation,
  pp.remove_diacritics,
  pp.remove_stopwords,
  pp.remove_whitespace,
  pp.remove_brackets
]

# Model scorers
REFIT_SCORE   = 'recall_score'
MODEL_SCORERS = {
  'precision_score': make_scorer(precision_score),
  'recall_score': make_scorer(recall_score),
  'accuracy_score': make_scorer(accuracy_score)
}

# GridSearchCV pipelines
MODEL_PIPELINES = {
    'fe=tfid+bigram+default_c=LR' : {
        'Pipeline' : Pipeline([
            ('tfid', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ]),
        'Parameters' : {
            'tfid__norm' : ['l1', 'l2'],
            'tfid__ngram_range': [(1, 2)],
            'clf__penalty': ['l2'],
            'clf__max_iter': [1000],
            'clf__C' : [0.25, 0.5, 1, 2, 4]
        }
    },
    'fe=tfid+trigram+default_c=LR' : {
        'Pipeline' : Pipeline([
            ('tfid', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ]),
        'Parameters' : {
            'tfid__norm' : ['l1', 'l2'],
            'tfid__ngram_range': [(1, 3)],
            'clf__penalty': ['l2'],
            'clf__max_iter': [1000],
            'clf__C' : [0.25, 0.5, 1, 2, 4]
        }
    }
}
