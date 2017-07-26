import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold
def cross_validation_score_statement(estimator,X,y,scoring,n_splits=5,statement=None,random_state=0):
    """
    Evaluate a score by cross-validation. 
    The fit method will be performed on the entire train subset at each iteration,
    the predict method and scoring will be performed only for objects from test subset where statement is True
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    X : pandas.DataFrame
        The data to fit.
    y : pandas.Series
        The target variable to try to predict.
    scoring : callable 
        The scoring function of signature scoring(y_true,y_pred).
    statement : boolean numpy.array of shape equal to y.shape
        The mask showing the objects we want to evaluate estimator on.
    n_splits : int
        Number of folds for cross-validation
    random_state : int
        Random_state for KFold and StratifiedKFold    
    
    Returns
    -----------
    scores : array of float, shape=(n_splits,)
    
    """
    if statement is None:
        cv = KFold(n_splits=n_splits,shuffle=True,random_state=random_state)
        cv_iter = list(cv.split(X, y))
    else:
        cv = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_state)
        cv_iter = list(cv.split(X, statement))
    scores = []
    
    for train, test in cv_iter:
        estimator.fit(X.iloc[train,:].values,y.iloc[train].values)
        if statement is not None:
            y_statement = y.iloc[test].loc[statement[test]]
            pred_statement = estimator.predict(X.iloc[test,:].loc[statement[test]].values)
        else:
            y_statement = y.iloc[test]
            pred_statement = estimator.predict(X.iloc[test,:].values)
        scores.append(scoring(y_statement,pred_statement))
    return np.array(scores)
def cross_validation_score_fit_subset(estimator,X,y,scoring,n_splits=5,statement=None,random_state=0):
    """
    Evaluate a score by cross-validation. 
    The fit method will be performed on the subset of train subset at each iteration where statement is True.
    The predict method and scoring will be performed on the entire test subset.
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    X : pandas.DataFrame
        The data to fit.
    y : pandas.Series
        The target variable to try to predict.
    scoring : callable 
        The scoring function of signature scoring(y_true,y_pred).
    statement : boolean numpy.array of shape equal to y.shape
        The mask showing the objects we want to fit estimator on.
    n_splits : int
        Number of folds for cross-validation
    random_state : int
        Random_state for KFold and StratifiedKFold    
    
    Returns
    -----------
    scores : array of float, shape=(n_splits,)
    
    """
    if statement is None:
        cv = KFold(n_splits=n_splits,shuffle=True,random_state=random_state)
        cv_iter = list(cv.split(X, y))
    else:
        cv = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_state)
        cv_iter = list(cv.split(X, statement))
    scores = []
    
    for train, test in cv_iter:
        if statement is not None:
            estimator.fit(X.iloc[train,:].values[statement[train]],y.iloc[train].values[statement[train]])
        else:
            estimator.fit(X.iloc[train,:].values,y.iloc[train].values)
        y_ = y.iloc[test]
        pred_ = estimator.predict(X.iloc[test,:].values)
        scores.append(scoring(y_,pred_))
    return np.array(scores)
