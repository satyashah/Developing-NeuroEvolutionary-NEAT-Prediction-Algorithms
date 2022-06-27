from matplotlib import ticker
import pandas as pd

def getTrainingData():

    trainDF = pd.read_csv("TrainTrigger.csv")

    profits = trainDF["profit"].values.tolist()
    days = trainDF["days"].values.tolist()

    trainDF = trainDF.drop(columns=["Unnamed: 0", "profit", "Ticker", "days"])

    trainDF["success"] = trainDF["success"].astype(int)

    train_features = trainDF.copy()
    train_labels = train_features.pop('success')

    train_features = train_features.values.tolist()
    train_labels = train_labels.values.tolist()

    return train_features, train_labels, profits, days

def getTestingData():

    testDF = pd.read_csv("TestTrigger.csv")

    profits = testDF["profit"].values.tolist()
    days = testDF["days"].values.tolist()

    testDF = testDF.drop(columns=["Unnamed: 0", "profit", "Ticker", "days"])

    testDF["success"] = testDF["success"].astype(int)

    test_features = testDF.copy()
    test_labels = test_features.pop('success')

    test_features = test_features.values.tolist()
    test_labels = test_labels.values.tolist()

    return test_features, test_labels, profits, days

# print(len(test_features[0])) = 21



def getLimitsTrainingData():

    trainDF = pd.read_csv("TrainLimit.csv")
    trainDF = trainDF[trainDF['success']]
    
    trainDF = trainDF.drop(columns=["Unnamed: 0", "Ticker", "days", "success"])

    train_features = trainDF.copy()
    train_labelsTP = train_features["TP%"]
    train_labelsSL = train_features["SL%"]

    train_features = train_features.drop(columns=["SL%", "TP%"])

    train_features = train_features.values.tolist()
    train_labelsTP = train_labelsTP.values.tolist()
    train_labelsSL = train_labelsSL.values.tolist()

    return train_features, train_labelsTP, train_labelsSL

def getLimitsTestingData():
    testDF = pd.read_csv("TestLimit.csv")
    testDF = testDF[testDF['success']]
    
    testDF = testDF.drop(columns=["Unnamed: 0", "Ticker", "days", "success"])

    test_features = testDF.copy()
    test_labelsTP = test_features["TP%"]
    test_labelsSL = test_features["SL%"]

    test_features = test_features.drop(columns=["SL%", "TP%"])

    test_features = test_features.values.tolist()
    test_labelsTP = test_labelsTP.values.tolist()
    test_labelsSL = test_labelsSL.values.tolist()

    return test_features, test_labelsTP, test_labelsSL

#print(len(list(train_features.columns))) #19
