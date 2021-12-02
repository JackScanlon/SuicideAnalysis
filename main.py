import model
import preprocess as pp

shouldPreprocess = False

if __name__ == '__main__':
  if shouldPreprocess:
    data = pp.getDataset('RedditDataSentimental.xlsx')
    data = pp.cleanDataset(data, True, True)

  # Get preprocessed dataset
  data = pp.getDataset('dataset.csv')
  data = model.minSubSample(data, False)

  # Indication for Logistic Regression model
  if False:
    model.validateModel(data) # not executed except for testing due to time required to evaluate

  # Parameter tweaking to trade accuracy for greater positive recall / sensitivity
  if False: 
    model.gridSearchParameters(data) # not executed except for testing due to time required to evaluate

  # Final training
  model.trainFinalModel(data)
