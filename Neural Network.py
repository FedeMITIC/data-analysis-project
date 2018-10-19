# Result: accuracy ~30%
# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def main():
    # Load the files for accuracy
    test_data_accuracy = pd.read_csv('./accuracy/test_data.csv')
    train_data_accuracy = pd.read_csv('./accuracy/train_data.csv')
    train_labels_accuracy = pd.read_csv('./accuracy/train_labels.csv')
    # test_data_logloss = pd.read_csv('./log-loss/test_data.csv')
    # train_data_logloss = pd.read_csv('./log-loss/train_data.csv')
    # train_labels_logloss = pd.read_csv('./log-loss/train_labels.csv')

    # Read the values
    train_labels_accuracy = train_labels_accuracy.values
    train_data_accuracy = train_data_accuracy.values
    test_data_accuracy = test_data_accuracy.values

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(train_data_accuracy,
                                                        train_labels_accuracy,
                                                        test_size=0.33,
                                                        random_state=42)

    # Scale the data before feeding them in the Neural Network
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    test_data_accuracy = scaler.transform(test_data_accuracy)

    # Instantiate and train the Neural Network: final loss => Iteration 83, loss = 0.00049346
    mlp = MLPClassifier(hidden_layer_sizes=(264, 264, 264), verbose=True, tol=0.00001)
    mlp.fit(x_train, np.ravel(y_train))

    # Outputs the predictions: array of 6544 (0 to 6543)
    predictions = mlp.predict(test_data_accuracy)

    class_names = ['Pop_Rock', 'Electronic', 'Rap', 'Jazz', 'Latin', 'RnB',
                   'International', 'Country', 'Reggae', 'Blues']
    for j in range(1, 11, 1):
        index_acc = np.where(predictions == j)
        print(f'{class_names[j - 1]}: {index_acc[0].size}')
    print(f'Last value: {predictions[len(predictions) - 1]}')
    id_num = range(1, len(predictions) + 1)
    df = pd.DataFrame({'Sample_id': id_num, 'Sample_label': predictions})
    df.to_csv('attempt.csv')


if __name__ == "__main__":
    main()
