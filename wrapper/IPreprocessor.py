from abc import ABCMeta, abstractmethod


class IPreprocessor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X) -> None:
        """train Preprocessor on given Dataframe

            Parameters:
            X: pandas Dataframe

            Returns:
            None

           """
        pass

    @abstractmethod
    def fit_transform(self, X):
        """train Preprocessor and apply it on given Dataframe

                    Parameters:
                    X: pandas Dataframe

                    Returns:
                    transformed x: Dataframe

                   """
        pass

    @abstractmethod
    def transform(self, X):
        """apply Preprocessor on given Dataframe

                    Parameters:
                    X: pandas Dataframe

                    Returns:
                    transformed x: Dataframe

                   """
        pass

    @abstractmethod
    def get_params(self):
        """return parametrs of preprocessor

                    Parameters:
                    None

                    Returns:
                    params: dict

                   """
        pass
