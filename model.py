import math
import pipeline as pl
import preprocess as pp
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pprint import pprint
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

def minSubSample(data, shouldShuffle=False):
  """
  minSubSample - prepares DataFrame by taking ?shuffled subsample from data, based on minimum size of groups, to balance classes

  :param data: pandas DataFrame
  :param shouldLemmatise: boolean, defaults False - determines whether groups are shuffled
  :return: returns DataFrame
  """
  # Get smallest sample group index
  minIndex = np.argmin(data['Suicidal'].value_counts())
  
  # Separate into distinct groups
  suicidalGroup = data[data['Suicidal'] == 1]
  notSuicidal   = data[data['Suicidal'] == 0]
  suicidalGroup = suicidalGroup.reset_index(drop=True)
  notSuicidal   = notSuicidal.reset_index(drop=True)

  # Shuffle if required
  if shouldShuffle:
    suicidalGroup = shuffle(suicidalGroup)
    notSuicidal   = shuffle(notSuicidal)

  # Sample groups by smallest group size
  numSamples = len([notSuicidal, suicidalGroup][minIndex])
  notSuicidal = notSuicidal[:numSamples]
  suicidalGroup = suicidalGroup[:numSamples]

  # Merge groups of data and shuffle
  data = shuffle(pd.concat([suicidalGroup, notSuicidal])).reset_index(drop=True)

  return data

def validateModel(data):
  """
  validateModel - tests validity and prints accuracy score of Logistic Regression model

  :param data: pandas DataFrame
  :return: None
  """
  # Tfid transformation
  tfidfVect = TfidfVectorizer()
  trainContentTfid = tfidfVect.fit_transform(data['Sentence'])

  # Fit model
  skf = StratifiedKFold(n_splits=5, shuffle=True)
  skf.get_n_splits(trainContentTfid, data['Suicidal'])

  LR = LogisticRegression()
  scores = []
  for train_index, test_index in skf.split(trainContentTfid, data['Suicidal']):
    x_train, x_test = trainContentTfid[train_index], trainContentTfid[test_index]
    y_train, y_test = data.iloc[train_index, 1], data.iloc[test_index, 1]

    LR.fit(x_train, y_train)

    y_pred = LR.predict(x_test)
    scores.append(round(metrics.accuracy_score(y_test,y_pred) * 100, 2))
    # print(metrics.classification_report(y_test, y_pred, target_names=['Not Suicidal', 'Suicidal']))

  # Print scores
  scores = [{'Split': i, 'Accuracy': x} for i, x in enumerate(scores)]
  print('____LR indication____\n')
  pprint(scores)
  print('_____________________')

def evaluateModel(pipeline, train_data, train_labels):
  """
  evaluateModel - evaluate pipeline model from pl

  :param pipeline: pipeline referring to MODEL_PIPELINES index
  :param train_data: x_train data
  :param train_labels: y_train data
  :return: return GridSearchCV dataframe
  """
  # Get the current pipeline and its parameters
  curPipeline = pl.MODEL_PIPELINES[pipeline]['Pipeline']
  curPipelineParams = pl.MODEL_PIPELINES[pipeline]['Parameters']

  # Grid search cv to evaluate the models for scorers
  gs_eval = GridSearchCV(
      curPipeline,
      curPipelineParams,
      scoring=pl.MODEL_SCORERS,
      refit=pl.REFIT_SCORE,
      error_score='raise',
      cv=StratifiedKFold(n_splits=10, random_state=None),
      n_jobs=-1
  )
  gs_eval = gs_eval.fit(train_data, train_labels)

  print(f'Best params for {pl.REFIT_SCORE}')
  print(gs_eval.best_params_)

  return gs_eval

def evaluateModels(train_data, train_labels):
  """
  evaluateModels - evaluate all models and parameters within pl.MODEL_PIPELINES

  :param train_data: x_train data
  :param train_labels: y_train data
  :return: array (names of pipelines), pd DataFrame (cv_results_), array (best_params_)
  """
  names = [ ]; results = [ ]; params = [ ]
  for curPipeline in pl.MODEL_PIPELINES:
      gs = evaluateModel(curPipeline, train_data, train_labels)
      names.append(curPipeline)
      results.append(pd.DataFrame(gs.cv_results_))
      params.append(gs.best_params_)
      
  return names, results, params

def gridSearchParameters(data):
  """
  gridSearchParameters - hyper parameter tweaking as per pl model pipelines to trade accuracy for higher sensitivity

  :param data: pandas DataFrame
  :return: None
  """
  names, results, params = evaluateModels(data.iloc[:, 0], data.iloc[:, 1])

  for i, x in enumerate(names):
    print('Test -->' + x)
    print('Results -->')
    print(results[i].head())

def trainFinalModel(data):
  """
  trainFinalModel - final training model with best parameters

  :param data: pandas DataFrame
  :return: None
  """
  # Transform
  tfidfVect = TfidfVectorizer(ngram_range=(1, 3), norm='l1')
  trainContentTfidf = tfidfVect.fit_transform(data['Sentence'])

  x_train, x_test, y_train, y_test = train_test_split(trainContentTfidf, data.iloc[:,1], test_size=0.2)

  # Fit model
  LR = LogisticRegression(C=4, max_iter=1000, penalty='l2')
  LR.fit(x_train, y_train)

  # Predict
  y_pred = LR.predict(x_test)
  print(f' Accuracy Score : {round(metrics.accuracy_score(y_test, y_pred) * 100, 2)}%')
  print(metrics.classification_report(y_test, y_pred))

  # Visualise model prediction vs truth
  sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot=True, fmt='d',cmap='YlGnBu')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()
