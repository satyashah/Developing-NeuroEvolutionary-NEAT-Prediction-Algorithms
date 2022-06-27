from matplotlib import ticker
import pandas as pd

def getTrainingData():

    trainDF = pd.read_csv("TrainNEAT.csv")

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

    testDF = pd.read_csv("TestNEAT.csv")

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

