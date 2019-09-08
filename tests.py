from wrapper.LogRegWrapper import LogRegWrapper
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # uncomment if you want to avoid warnings


class TestPreprocessor(unittest.TestCase):
    def test_missed_data(self):
        lg = LogRegWrapper()
        X = pd.DataFrame([[-4, 1], [0.4, 2], [40, None], [4, 23]], columns=['obj', 'numbers'])
        X_expected = pd.DataFrame([[-4, 1], [0.4, 2], [40, -1], [4, 23]], columns=['obj', 'numbers'])
        assert_frame_equal(lg._preprocessor._handling_missed_value(X), X_expected, check_dtype=False)

    def test_label_encoder(self):
        lg = LogRegWrapper()
        X_train = pd.DataFrame(['a', 'v', 'c', 'v'], columns=['obj'], dtype='object')
        X_test = pd.DataFrame(['f', 'v', 'c', 'u'], columns=['obj'], dtype='object')
        lg._preprocessor.fit_transform(X_train)
        X_test_res = lg._preprocessor.transform(X_test)
        X_test_expected = pd.DataFrame([-1, 2, 3, -1], columns=['obj'], dtype='int')
        assert_frame_equal(X_test_expected, X_test_res, check_dtype=False)


class TestWrapper(unittest.TestCase):

    def test_no_pretrained_predict(self):
        lg = LogRegWrapper()
        self.assertRaises(Exception, lg.predict, self.X)

    def test_no_pretrained_predict_proba(self):
        lg = LogRegWrapper()
        self.assertRaises(Exception, lg.predict_proba, self.X)

    def test_no_pretrained_evaluate(self):
        lg = LogRegWrapper()
        self.assertRaises(Exception, lg.evaluate, self.X, self.y)

    def test_reprodusible(self):
        lg1 = LogRegWrapper()
        lg1.fit(self.X, self.y)
        res1 = lg1.predict_proba(self.X)

        lg2 = LogRegWrapper()
        lg2.fit(self.X, self.y)
        res2 = lg2.predict_proba(self.X)

        np.testing.assert_equal(res1, res2)

    def test_output_format_fit(self):
        lg = LogRegWrapper()
        self.assertIsNone(lg.fit(self.X, self.y))

    def test_output_format_predict(self):
        lg = LogRegWrapper()
        lg.fit(self.X, self.y)
        y_pred = lg.predict(self.X)

        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(y_pred.shape, lg.predict(self.X).shape)
        self.assertSetEqual(set(y_pred), {0, 1})

    def test_output_format_predict_proba(self):
        lg = LogRegWrapper()
        lg.fit(self.X, self.y)
        y_pred = lg.predict_proba(self.X)

        self.assertIsInstance(y_pred, np.ndarray)
        self.assertEqual(self.y.shape[0], y_pred.shape[0])
        self.assertEqual(len(set(self.y)), y_pred.shape[1])

    def test_output_format_evaluate(self):
        lg = LogRegWrapper()
        lg.fit(self.X, self.y)
        res = lg.evaluate(self.X, self.y)

        self.assertIsInstance(res, dict)
        self.assertIn('f1_score', res)
        self.assertIn('logloss', res)

    def test_output_format_tume_parametrs(self):
        lg = LogRegWrapper()
        res = lg.tune_parameters(self.X, self.y)

        self.assertIsInstance(res, dict)
        self.assertIn('scores', res)
        self.assertIsInstance(res['scores'], dict)
        self.assertIn('f1_score', res['scores'])
        self.assertIn('logloss', res['scores'])

    def setUp(self) -> None:
        C1 = np.array([[0., -0.8], [1.5, 0.8]])
        C2 = np.array([[1., -0.7], [2., 0.7]])
        gauss1 = np.dot(np.random.randn(200, 2) + np.array([5, 3]), C1)
        gauss2 = np.dot(np.random.randn(200, 2) + np.array([1.5, 0]), C2)

        X = np.vstack([gauss1, gauss2])
        self.X = pd.DataFrame(data=X, columns=['X1', 'X2'])
        self.y = np.r_[np.ones(200), np.zeros(200)]


if __name__ == '__main__':
    unittest.main()
