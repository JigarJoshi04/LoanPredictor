import pandas as pd
import numpy as np
from sklearn import model_selection

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

def create_dataframe_from_csv(filepath):
    """
    creating dataframe from csv
    """
    df = pd.read_csv(filepath)
    return df

def split_data(df):
    """
    splits the datframe into train and test data
    train data and test data are further splitted to feature and labels
    """
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :8], df.iloc[:, 8:],test_size=0.15, random_state=42)
    return x_train, x_test, y_train, y_test

def train_model(x_train, y_train):
    """
    training the random forest classifier model using x_train and y_train
    """
    model = RandomForestClassifier(n_estimators=50)
    print("INFO:    Model training started")
    model.fit(x_train, y_train)
    print("INFO:    Model training ended")
    return model

def score_model(x_test, y_test, model):
    """
    find the accuracy of the model using the x_test and y_test
    """
    accuracy = model.score(x_test, y_test)
    print("INFO:    Accuracy of the model is  ", round(accuracy*100,2))
    return accuracy

def get_prediction(labels, model):
    """
    get the predictions from the model for x_test
    """
    predictions = model.predict(labels)
    return predictions

def error_calculation(predictions, y_test):
    """
    get the MSE  error calculated between the predictions and the actual data
    """
    mse = mean_squared_error(y_test, predictions)
    print("INFO:    Mean square error is : ", round(mse, 2))
    return mse

def main():
    """
    driver function to get the prediction
    """
    filepath  = "all_data.csv"
    main_df = create_dataframe_from_csv(filepath)
    questioned_df = main_df[:][:700]
    answer_df = main_df[:][700:]

    x_train, x_test, y_train, y_test = split_data(questioned_df)
    model = train_model(x_train, y_train)
    score_model(x_test, y_test, model)

    predictions = get_prediction(x_test, model)
    error_calculation(predictions, y_test)

    answer_predictions = get_prediction(answer_df.iloc[:, :8], model)
    answer_df["Predictions"] = answer_predictions
    answer_df.to_csv("predicted.csv")
    print("INFO:    Output file is created")

main()

