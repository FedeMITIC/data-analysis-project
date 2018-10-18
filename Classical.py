# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


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
    # test_data_logloss = pd.read_csv('./log-loss/test_data.csv')
    # train_data_logloss = pd.read_csv('./log-loss/train_data.csv')
    # train_labels_logloss = pd.read_csv('./log-loss/train_labels.csv')
    test_data_accuracy = pd.read_csv('./accuracy/test_data.csv')
    train_data_accuracy = pd.read_csv('./accuracy/train_data.csv')
    train_labels_accuracy = pd.read_csv('./accuracy/train_labels.csv')

    test_data_accuracy = test_data_accuracy.values
    test_data_resized = np.empty((len(test_data_accuracy), 4))
    for i in range(test_data_accuracy.shape[0]):
        # ID
        test_data_resized[i, 0] = test_data_accuracy[i, 0]
        # Rhythm mean
        test_data_resized[i, 1] = np.mean(test_data_accuracy[i, 1:169])
        # Chroma mean
        test_data_resized[i, 2] = np.mean(test_data_accuracy[i, 170:217])
        # MFCCs mean
        test_data_resized[i, 3] = np.mean(test_data_accuracy[i, 218:265])
    print(test_data_resized)
    # Reduce the dimensions of the test_data to avoid overfitting
    # test_data_accuracy = train_test_split(test_data_accuracy)

    # for i in range(len(predictions)):
    #     print(f'{i}: {predictions[i]}')
    # for j in range(1, 11, 1):
    #     index = np.where(predictions == j)
    #     print(f'{j}: {index[0].size}')
    # id_num = range(1, len(predictions) + 1)
    # df = pd.DataFrame({'Sample_id': id_num, 'Sample_label': predictions})
    # df.columns = ['Sample_id', 'Sample_label']
    # df.to_csv('attempt.csv')


if __name__ == "__main__":
    main()
