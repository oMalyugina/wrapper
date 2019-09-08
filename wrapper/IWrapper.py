from abc import ABCMeta, abstractmethod


class IWrapper:
    __metaclass__ = ABCMeta

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
