from wrapper_interface import Wrapper
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class LogRegWrapper(Wrapper):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(class_weight='balanced', random_state=123)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        self.predict(X)

    def predict_proba(self, X):
        self.model.predict_proba(X)

    def evaluate(self, X, y):
        self.model.score(X, y)

    def tune_parameters(self, X, y):
        tuned_parameters = {'penalty': ('l1', 'l2', 'elasticnet', 'none'), 'dual': [False, True],
                      'tol':[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.5, 1], 'C':[0.1, 0.5, 1, 5, 10],
                      'fit_intercept':[True, False], 'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                      'max_iter':[100,1000,2000,5000]}
        clf = GridSearchCV(self.model, tuned_parameters, cv=5)
        clf.fit(X, y)
        sorted(clf.cv_results_.keys())

