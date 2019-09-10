from sklearn.preprocessing import Normalizer
import category_encoders as ce
from wrapper.IPreprocessor import IPreprocessor


class Preprocessor_X(IPreprocessor):

    def __init__(self):
        self._label_encoders = {}
        self._normalizers = {}
        self._missed_ratio_to_delete = 10
        self._trained = False

    def fit(self, X_original):
        self._trained = False
        X = X_original.copy()
        X = self._label_encoding(X)
        X = self._handling_missed_value(X)
        X = self._normalized(X)
        self._trained = True

    def fit_transform(self, X_original):
        self._trained = False
        X = X_original.copy()
        X = self._label_encoding(X)
        X = self._handling_missed_value(X)
        X = self._normalized(X)
        self._trained = True
        return X

    def transform(self, X_original):
        if not self._trained:
            raise Exception("train Preprocessor before using transformation")
        X = X_original.copy()
        X = self._label_encoding(X)
        X = self._handling_missed_value(X)
        X = self._normalized(X)
        return X

    def get_params(self) -> dict:
        if not self._trained:
            raise Exception("train Preprocessor before getting parameters")
        params = {}
        params["missed_ratio_to_delete"] = self._missed_ratio_to_delete
        for name, encoders_dict in zip(["label_encoding", "normalized"], [self._label_encoders, self._normalizers]):
            encoders_param = {}
            for var_name, encoder in encoders_dict:
                encoders_param[var_name] = encoder.get_params()
            params[name] = encoders_param

        return params

    def _handling_missed_value(self, X):
        all_data_na = (X.isnull().sum() / len(X)) * 100
        for name, ratio in all_data_na.items():
            if len(X.columns) < 2:
                print("only one good feature or to many missed values")
                return X.fillna(-1)
            if ratio > self._missed_ratio_to_delete:
                X.drop(name, axis=1)
        return X.fillna(-1)

    def _normalized(self, X):
        if self._trained:
            for var_name in X.select_dtypes(include=['object']):
                X[var_name] = self._normalizers[var_name].transform(X[var_name])
        else:
            self._normalizers = {}
            for var_name in X.select_dtypes(include=['object']):
                self._normalizers[var_name] = Normalizer()
                X[var_name] = self._normalizers[var_name].fit_transform(X[var_name])
        return X

    def _label_encoding(self, X):
        if self._trained:
            for var_name in X.select_dtypes(include=['object']):
                X[var_name] = self._label_encoders[var_name].transform(X[var_name].values.astype(str))
        else:
            self._label_encoders = {}
            for var_name in X.select_dtypes(include=['object']):
                self._label_encoders[var_name] = ce.OrdinalEncoder()
                X[var_name] = self._label_encoders[var_name].fit_transform(X[var_name].values.astype(str))
        return X
