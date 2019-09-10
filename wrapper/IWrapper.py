from abc import ABCMeta, abstractmethod


class IWrapper:
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X, y) -> None:
        """Fit the model according to the given training data.

        Parameters:
        X: pandas.DataFrame, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y: numpy.array, shape (n_samples,)
            Target vector relative to X.

        Returns:
        None

       """
        pass

    @abstractmethod
    def predict(self, X):
        """predict class for given objects

        Parameters:
        X: pandas.DataFrame, shape = [n_samples, n_features]

        Returns:
        y: numpy.array, shape = [n_samples]
        Returns the predicted label.


        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters:
        X: pandas.DataFrame, shape = [n_samples, n_features]

        Returns:
        y: numpy.array, shape = [n_samples, n_classes]
        Returns the probability of the sample for each class in the model,
        where classes are ordered as they are in ``self.classes_``.

           """

    pass

    @abstractmethod
    def evaluate(self, X, y):

        """Returns the score (f1 and log_loss) on the given test data and labels.

        Parameters:
        X: pandas.DataFrame, shape = (n_samples, n_features)
            Test samples.
        y: numpy.array, shape = (n_samples,)
            True labels for X.

        Returns:
            scores: dict containing "f1_score" and "logloss" keys


       """

    pass

    @abstractmethod
    def tune_parameters(self, X, y):
        """find the best parameters for given data and fit model with the parameters

        X: pandas.DataFrame, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y: numpy.array, shape (n_samples,)
            Target vector relative to X.

        Returns:
            results: dict containing best parameters and average scores


       """
        pass
