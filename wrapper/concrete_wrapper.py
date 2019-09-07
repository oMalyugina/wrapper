from wrapper.wrapper_interface import Wrapper
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.metrics import f1_score, log_loss
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)  # uncomment if you want to avoid warnings


class LogRegWrapper(Wrapper):
    def __init__(self):
        super().__init__()
        self._model = LogisticRegression(class_weight='balanced', random_state=123)

    def _label_encoding(self, X):
        label_encoder = LabelEncoder()
        for var_name in X.select_dtypes(include=['object']):
            X[var_name] = label_encoder.fit_transform(X[var_name].astype(str))
        return X

    def _preprossecing_X(self, X):
        X = self._label_encoding(X)
        X = self._handling_missed_value(X)
        X = self._normalized(X)
        X = self._feature_selection(X)
        return X

    def fit(self, X, y) -> None:
        X = self._preprossecing_X(X)
        self._model.fit(X, y)

    def predict(self, X):
        X = self._preprossecing_X(X)
        return self._model.predict(X)

    def predict_proba(self, X):
        X = self._preprossecing_X(X)
        return self._model.predict_proba(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {'f1_score': f1_score(y, y_pred), 'logloss': log_loss(y, y_pred)}

    def tune_parameters(self, X, y):
        X = self._preprossecing_X(X)
        tuned_parameters = {  # 'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.5, 1], 'C': [0.1, 0.5, 1, 5, 10],
            # 'fit_intercept': [True, False],
            # 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [2, 5, 10, 100, 1000]}

        # tuned_parameters = {'penalty': ('l1', 'l2', 'elasticnet', 'none'), 'dual': [False, True],
        #                     'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.5, 1], 'C': [0.1, 0.5, 1, 5, 10],
        #                     'fit_intercept': [True, False],
        #                     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        #                     'max_iter': [100, 1000, 2000, 5000]}
        clf = GridSearchCV(self._model, tuned_parameters, cv=5, iid=True, scoring=['accuracy', 'f1', 'neg_log_loss'],
                           refit='accuracy')
        clf.fit(X, y)
        self._model = clf.best_estimator_
        cv_res = pd.DataFrame(clf.cv_results_)
        res = clf.best_estimator_.get_params()
        scores = {'f1_score':cv_res['mean_test_f1'].mean(), 'logloss':cv_res['mean_test_neg_log_loss'].mean()}
        res['scores'] = scores
        return res

    def _handling_missed_value(self, X):
        all_data_na = (X.isnull().sum() / len(X)) * 100
        for name, ratio in all_data_na.items():
            if ratio > 10:
                X.drop(name, axis=1)
        return X.fillna(-1)

    def _normalized(self, X):
        scaler = Normalizer()
        return scaler.fit_transform(X)

    def _feature_selection(self, X):
        return X


if __name__ == '__main__':
    path_to_data = '../DR_Demo_Lending_Club_reduced.csv'
    data = pd.read_csv(path_to_data)
    y = data.is_bad.values
    X = data.drop('is_bad', axis=1)
    X.head()
    lg = LogRegWrapper()
    lg.fit(X, y)
