from wrapper.concrete_wrapper import LogRegWrapper
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


class TestWrapper(unittest.TestCase):

    def test_missed_data(self):
        lg = LogRegWrapper()
        X = pd.DataFrame([[-4, 1], [0.4, 2], [40, None], [4, 23]], columns=['obj', 'numbers'])
        X_expected = pd.DataFrame([[-4, 1], [0.4, 2], [40, -1], [4, 23]], columns=['obj', 'numbers'])
        assert_frame_equal(lg._handling_missed_value(X), X_expected, check_dtype=False)

    def test_exceptions(self):
        lg = LogRegWrapper()
        with self.assertRaises(Exception) as cm:
            lg.predict(self.X)
        the_exception = cm.exception
        self.assertEqual(str(the_exception), "model wasn't trained")
        lg = LogRegWrapper()
        with self.assertRaises(Exception) as cm:
            lg.predict_proba(self.X)
        the_exception = cm.exception
        self.assertEqual(str(the_exception), "model wasn't trained")
        lg = LogRegWrapper()
        with self.assertRaises(Exception) as cm:
            lg.evaluate(self.X, self.y)
        the_exception = cm.exception
        self.assertEqual(str(the_exception), "model wasn't trained")

    def test_label_encoder(self):
        lg = LogRegWrapper()
        X_train = pd.DataFrame(['a', 'v', 'c', 'v'], columns=['obj'], dtype='object')
        y_train = [1, 1, 0, 0]
        X_test = pd.DataFrame(['f', 'v', 'c', 'u'], columns=['obj'], dtype='object')
        y_test = [1, 1, 0, 1]
        lg.fit(X_train, y_train)
        lg.evaluate(X_test, y_test)

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
        self.assertIsInstance(lg.predict(self.X), np.ndarray)
        self.assertEqual(self.y.shape, lg.predict(self.X).shape)
        self.assertSetEqual(set(lg.predict(self.X)), {0, 1})

    def test_output_format_predict_proba(self):
        lg = LogRegWrapper()
        lg.fit(self.X, self.y)
        self.assertIsInstance(lg.predict_proba(self.X), np.ndarray)
        self.assertEqual(self.y.shape[0], lg.predict_proba(self.X).shape[0])
        self.assertEqual(len(set(self.y)), lg.predict_proba(self.X).shape[1])

    def test_output_format_evaluate(self):
        lg = LogRegWrapper()
        lg.fit(self.X, self.y)

        self.assertIsInstance(lg.evaluate(self.X, self.y), dict)
        self.assertIn('f1_score', lg.evaluate(self.X, self.y))
        self.assertIn('logloss', lg.evaluate(self.X, self.y))

    def test_output_format_tume_parametrs(self):
        lg = LogRegWrapper()

        self.assertIsInstance(lg.tune_parameters(self.X, self.y), dict)
        self.assertIn('scores', lg.tune_parameters(self.X, self.y))
        self.assertIsInstance(lg.tune_parameters(self.X, self.y)['scores'], dict)
        self.assertIn('f1_score', lg.tune_parameters(self.X, self.y)['scores'])
        self.assertIn('logloss', lg.tune_parameters(self.X, self.y)['scores'])

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
