# Result: accuracy ~30%
# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def main():
    test_data_accuracy = pd.read_csv('./accuracy/test_data.csv')
    train_data_accuracy = pd.read_csv('./accuracy/train_data.csv')
    train_labels_accuracy = pd.read_csv('./accuracy/train_labels.csv')

    train_labels_accuracy = train_labels_accuracy.values
    train_data_accuracy = train_data_accuracy.values
    test_data_accuracy = test_data_accuracy.values

    # test_data_logloss = pd.read_csv('./log-loss/test_data.csv')
    # train_data_logloss = pd.read_csv('./log-loss/train_data.csv')
    # train_labels_logloss = pd.read_csv('./log-loss/train_labels.csv')

    # Scales the training data
    scaler = StandardScaler()
    scaler.fit(train_data_accuracy)
    scaler.transform(train_data_accuracy)
    scaler.transform(test_data_accuracy)

    # Train the model for the accuracy challenge
    mlp = MLPClassifier(hidden_layer_sizes=(264))
    mlp.fit(train_data_accuracy, np.ravel(train_labels_accuracy))
    # predictions = predict(train_data_accuracy)
    predictions = mlp.predict(test_data_accuracy)
    data = np.empty((10, 2), dtype=int)
    class_names = ['Pop_Rock', 'Electronic', 'Rap', 'Jazz', 'Latin', 'RnB',
                   'International', 'Country', 'Reggae', 'Blues']
    for j in range(1, 11, 1):
        index = np.where(predictions == j)
        data[j - 1, 1] = j
        data[j - 1, 0] = index[0].size
        print(f'{class_names[j - 1]}: {index[0].size}')
    id_num = range(1, len(predictions) + 1)
    df = pd.DataFrame({'Sample_id': id_num, 'Sample_label': predictions})
    df.columns = ['Sample_id', 'Sample_label']
    df.to_csv('attempt.csv')


if __name__ == "__main__":
    main()
