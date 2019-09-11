### Logistic Regression Wrapper
#### Description
I wrote a Wrapper for regularized logistic regression. 

I rewrote interface from the task description as abstract IWrapper class 
which is implemented with the class LogRegWrapper.

I suggest that input data is correct so I don't check them.

Also, I moved the data preprocessing into a separate class which implements interface IPreprocessor. 
An instance of the preprocessor is injected into Wrapper via constructor and
 potentially might be replaced with any other implementation.
The parameters for Logistic Regression can be also provided to LogRegWrapper 
in the constructor's parameter in a dictionary.

Usage of the Wrapper you can see in file ```testWrapper.ipynb```

#### Run tests
##### Run tests locally
To be able to run the project you should have python3.

To install all the required dependencies locally you may run the command:
```
pip3 install -r requirements.txt
```



All unit tests described in the task can be found in the file ```tests.py```.

You can run it from a terminal with the command: 
```
python -m unittest tests
```

##### Run tests in Docker

If you have Docker on your computer test can be ran with the next commands:
```
docker build -t wrapper-tests .
docker run --rm wrapper-tests
```

#### Short-Answer Portion

1. I will recommend a logistic regression. 
Firstly, the model more tractable. 
You can define weight as features' influence (if data was normalized).
Secondly, a neural network can be overfitted with larger probability.
If we have the same quality using simpler model is better.

2. In my opinion, the easiest way is using logistic regression weights like an analog of correlation.
And they can visualize the forest's trees.