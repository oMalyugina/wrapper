from wrapper.IWrapper import IWrapper
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, log_loss
import warnings
from wrapper.Preprocessor import Preprocessor_X

warnings.filterwarnings("ignore", category=FutureWarning)  # uncomment if you want to avoid warnings


class LogRegWrapper(IWrapper):
    def __init__(self, preprocessor=Preprocessor_X(), log_reg_params=None):
        super().__init__()
        if log_reg_params is None:
            log_reg_params = {'class_weight': 'balanced', 'random_state': 123}
        self._model = LogisticRegression(**log_reg_params)
        self._preprocessor = preprocessor
        self._trained = False

    def fit(self, X, y) -> None:
        # TODO написать докстринги
        self._trained = False
        X = self._preprocessor.fit_transform(X)
        self._model.fit(X, y)
        self._trained = True

    def predict(self, X):
        X = self._preprocessor.transform(X)
        return self._model.predict(X)

    def predict_proba(self, X):
        X = self._preprocessor.transform(X)
        return self._model.predict_proba(X)

    def evaluate(self, X, y):
        X = self._preprocessor.transform(X)
        y_pred = self.predict(X)
        return {'f1_score': f1_score(y, y_pred), 'logloss': log_loss(y, y_pred)}

    def tune_parameters(self, X, y):
        self._trained = False
        X = self._preprocessor.fit_transform(X)
        # TODO подумать как сделать полный перебор парамервов
        tuned_parameters = {  # 'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.5, 1], 'C': [0.1, 0.5, 1, 5, 10],
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
        res = clf.best_estimator_.get_params()
        cv_res = pd.DataFrame(clf.cv_results_)
        scores = {'f1_score': cv_res['mean_test_f1'].mean(), 'logloss': cv_res['mean_test_neg_log_loss'].mean()}
        res['scores'] = scores
        self._trained = True
        return res


if __name__ == '__main__':
    # TODO создать инструкцию как запускать
    path_to_data = '../DR_Demo_Lending_Club_reduced.csv'
    data = pd.read_csv(path_to_data)
    y = data.is_bad.values
    X = data.drop('is_bad', axis=1)
    X.head()
    lg = LogRegWrapper()
    lg.fit(X, y)
