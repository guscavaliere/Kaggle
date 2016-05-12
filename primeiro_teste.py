# -*- coding: utf-8 -*-
"""
Created on Wed May 11 21:05:30 2016

@author: Gustavo
"""

import pandas as pd
# read the data
titanic =pd.read_csv('train.csv')
# print to see the description of the data
print(titanic.describe())
# filling the missing values on age column
titanic['Age']=titanic['Age'].fillna(titanic['Age'].median())
print (titanic.describe())
#replace all male values for 0 and female values for 1, as we can't read string labels
titanic.loc[titanic['Sex']=='male','Sex']=0
titanic.loc[titanic['Sex']=='female','Sex']=1
# print to see if it works
print (titanic['Sex'].unique())

#setting up the embarked labesl
titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
print(titanic["Embarked"].unique())

#let's print and see some of our data
# print(titanic.head())
#--------------------------------------------------------------------------#
#copied this part from dataquest
# Regressão Linear
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)
#----------------------------------------------------------------------------#    
import numpy as np

# The predictions are in three separate numpy arrays.  Concatenate them into one.  
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print ("Score de Regressão Linear ", accuracy)

#-------------------------------------------------------------------------#
#Regressão Logística
import sklearn
from sklearn import cross_validation


# Initialize our algorithm
alg = sklearn.linear_model.LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print("Score de Regressão Logística ",scores.mean())


#--------------------------------------------------------------------#

# Initialize the algorithm class
alg = sklearn.linear_model.LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": titanic["PassengerId"],
        "Survived": predictions
    })
    
submission.to_csv("kaggle.csv", index=False)