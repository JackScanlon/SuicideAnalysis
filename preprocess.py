import os
import emoji
import nltk
import pipeline as pl
import texthero as hero
import pandas as pd
import regex as re
from nltk.stem import WordNetLemmatizer

# Utilities
def getDataset(filename):
  """
  getDataset - gets and reads the specified file from the input directory,
               drops null rows and columns excluded from our analysis.

  :param filename: in format of {filename.extension}
  :return: returns pandas DataFrame
  """ 
  # Read file
  filename = pl.INPUT_DIRECTORY + filename
  split = os.path.splitext(filename)
  data  = None
  if 'csv' in split[1]:
    data = pd.read_csv(filename)
  elif 'xlsx' in split[1]:
    data = pd.read_excel(filename)

  # Select only Suicidal and Sentence columns
  data = data[['Sentence', 'Suicidal']]

  # Remove null rows
  data = data.dropna()

  return data

def removeHandles(text):
  """
  removeHandles - removes @alphanumeric handles from text

  :param text: string
  :return: returns cleansed string
  """ 
  text = ' '.join(re.sub('@\w+', '', text).split())
  return text

def removeHashtag(text):
  """
  removeHashtag - removes #hashtag from text

  :param text: string
  :return: returns cleansed string
  """ 
  text = ' '.join(re.sub('\#[\w\_]+', '', text).split())
  return text

def removeEmoji(text):
  """
  removeEmoji - removes all instances of emoji in the string passed.

  :param text: string
  :return: returns cleansed string
  """ 
  text = emoji.get_emoji_regexp().sub(u'',text)
  return text

def cleanSentences(text, lemmatise=True):
  """
  cleanSentences - pre-processes text for analysis, optional lemmatisation (defaults to True)

  :param text: array of strings
  :param lemmatise: boolean, defaults True - determines whether function performs lemmatisation
  :return: returns array of cleaned strings
  """ 
  # Remove emoji
  text = text.apply(lambda x: ' '.join([removeEmoji(word) for word in x.split()]))

  # Remove hashtag
  text = text.apply(lambda x: ' '.join([removeHashtag(word) for word in x.split()]))

  # Remove handles
  text = text.apply(lambda x: ' '.join([removeHandles(word) for word in x.split()]))

  # Clean text as per the pipeline
  text = hero.clean(text, pipeline=pl.PREPROCESSING_PIPELINE)

  # Lemmatisation
  if lemmatise:
    nltk.download('wordnet')
    text = text.apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(word) for word in x.split()]))
  
  # Remove words with < 2 len and/or aren't alphabetical
  text = text.apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2 and word.isalpha()]))

  return text

def cleanDataset(data, shouldLemmatise=True, shouldSave=False):
  """
  cleanDataset - prepares DataFrame for analysis

  :param data: pandas DataFrame
  :param shouldLemmatise: boolean, defaults to True - passed to cleanSentences if lemmatisation required
  :param shouldSave: boolean, defaults to False - determines whether we save the cleaned DF 
                                                  as a csv to ./inputs/dataset.csv
  :return: returns cleansed DataFrame
  """
  # Replace yes/no string datapoint as binary representation
  data.loc[data['Suicidal'] == 'Yes', 'Suicidal'] = 1
  data.loc[data['Suicidal'] == 'No', 'Suicidal']  = 0

  # Preprocess string data
  data['Sentence'] = cleanSentences(data['Sentence'], shouldLemmatise)

  # Save input if required
  if shouldSave:
    data.to_csv(pl.INPUT_DIRECTORY + 'dataset.csv')
  
  return data
