# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def display_data(data, name):
    print(f'{name}:\n')
    print(data)
    print('\n')


def normalize_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)


def predict(data):
    return mlp.predict(data)


def train_accuracy_challenge(train_data, train_labels):
    # Expose the network globally
    global mlp
    # Train the model using 3 layers: each layer has a number of neurons that equal the features
    mlp = MLPClassifier(hidden_layer_sizes=264)
    mlp.fit(train_data, train_labels)


def main():
    test_data_accuracy = pd.read_csv('./accuracy/test_data.csv')
    train_data_accuracy = pd.read_csv('./accuracy/train_data.csv')
    train_labels_accuracy = pd.read_csv('./accuracy/train_labels.csv')

    # reduce_features(test_data_accuracy)

    # print(test_data_accuracy[0].shape)
    print(len(test_data_accuracy))
    print(len(train_data_accuracy))
    print(len(train_labels_accuracy))

    # test_data_logloss = pd.read_csv('./log-loss/test_data.csv')
    # train_data_logloss = pd.read_csv('./log-loss/train_data.csv')
    # train_labels_logloss = pd.read_csv('./log-loss/train_labels.csv')

    # display_data(test_data_accuracy, 'Test data accuracy')
    # display_data(test_data_logloss, 'Test data logloss')
    # display_data(train_data_accuracy, 'Train data accuracy')
    # display_data(train_labels_accuracy, 'Train label accuracy')
    # display_data(train_data_logloss, 'Train data logloss')
    # display_data(train_labels_logloss, 'Train label logloss')

    # Scales the training data
    train_data_accuracy = normalize_data(train_data_accuracy)
    test_data_accuracy = normalize_data(test_data_accuracy)

    # Train the model for the accuracy challenge
    train_accuracy_challenge(train_data_accuracy, train_labels_accuracy)
    # predictions = predict(train_data_accuracy)
    predictions = predict(test_data_accuracy)
    for i in range(len(predictions)):
        print(f'{i}: {predictions[i]}')
    for j in range(1, 11, 1):
        index = np.where(predictions == j)
        print(f'{j}: {index[0].size}')
    id_num = range(1, len(predictions) + 1)
    df = pd.DataFrame({'Sample_id': id_num, 'Sample_label': predictions})
    df.columns = ['Sample_id', 'Sample_label']
    df.to_csv('attempt.csv')


if __name__ == "__main__":
    main()
