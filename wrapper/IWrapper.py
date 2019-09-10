from abc import ABCMeta, abstractmethod


class IWrapper:
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X, y) -> None:
        """train model on given data

                    Parameters:
                    X: pandas.DataFrame
                    y: numpy.array

                    Returns:
                    None

                   """
        pass

    @abstractmethod
    def predict(self, X):
        """predict class for given objects

                    Parameters:
                    X: pandas.DataFrame

                    Returns:
                    y: numpy.array

                   """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """predict probability for each class for given objects

            Parameters:
            X: pandas.DataFrame

            Returns:
            y: numpy.array

           """

    pass

    @abstractmethod
    def evaluate(self, X, y):

        """predict class for given objects and compute quality

        Parameters:
        X: pandas.DataFrame
        y: numpy.array

        Returns:
            scores: dict


       """

    pass

    @abstractmethod
    def tune_parameters(self, X, y):
        """tune parameters of classifier

        Parameters:
        X: pandas.DataFrame
        y: numpy.array

        Returns:
            best parameters and average scores: dict


       """
        pass
