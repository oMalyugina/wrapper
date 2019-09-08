from abc import ABCMeta, abstractmethod


class Wrapper:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self._model = None
        self._trained = False
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass

    @abstractmethod
    def tune_parameters(self, X, y):
        pass
