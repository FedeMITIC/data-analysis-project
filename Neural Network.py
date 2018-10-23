# Result: accuracy ~30%
# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def main():
    # Load the files for accuracy
    # test_data_accuracy = pd.read_csv('./accuracy/test_data.csv')
    # train_data_accuracy = pd.read_csv('./accuracy/train_data.csv')
    # train_labels_accuracy = pd.read_csv('./accuracy/train_labels.csv')
    test_data_accuracy = pd.read_csv('./accuracy/new_test_data.csv')
    train_data_accuracy = pd.read_csv('./accuracy/new_train_data.csv')
    train_labels_accuracy = pd.read_csv('./accuracy/train_labels.csv')
    # test_data_logloss = pd.read_csv('./log-loss/test_data.csv')
    # train_data_logloss = pd.read_csv('./log-loss/train_data.csv')
    # train_labels_logloss = pd.read_csv('./log-loss/train_labels.csv')

    for i in range(1, 11, 1):
        print(f'Label {i}: {len(np.where(train_labels_accuracy.values == i)[0])}')

    # Read the values
    train_labels_accuracy = train_labels_accuracy.values
    train_data_accuracy = train_data_accuracy.values
    test_data_accuracy = test_data_accuracy.values

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(train_data_accuracy,
                                                        train_labels_accuracy,
                                                        test_size=0.1,
                                                        random_state=42)

    # # Scale the data before feeding them in the Neural Network
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    test_data_accuracy = scaler.transform(test_data_accuracy)

    # Instantiate and train the Neural Network: final loss => Iteration 166, loss = 0.00032670
    mlp = MLPClassifier(activation='relu',
                        alpha=0.0001,
                        hidden_layer_sizes=(100, 100),
                        learning_rate='constant',
                        max_iter=1000,
                        shuffle=True,
                        solver='adam',
                        tol=0.000001,
                        verbose=True
                        )
    mlp.fit(x_train, np.ravel(y_train))

    # Outputs the predictions: array of 6544 (0 to 6543)
    predictions = mlp.predict(test_data_accuracy)
    print(f'Total predictions: {len(predictions)}')

    class_names = ['Pop_Rock', 'Electronic', 'Rap', 'Jazz', 'Latin', 'RnB',
                   'International', 'Country', 'Reggae', 'Blues']
    for j in range(1, 11, 1):
        index_acc = np.where(predictions == j)
        print(f'{class_names[j - 1]}: {index_acc[0].size} ({round((index_acc[0].size / len(predictions)) * 100, 2)}%)')
    print(f'Last value: {predictions[len(predictions) - 1]}')
    id_num = range(1, len(predictions) + 1)
    df = pd.DataFrame({'Sample_id': id_num, 'Sample_label': predictions}, columns=['Sample_id', 'Sample_label'])
    df.to_csv('attempt.csv')
    print(mlp.score(x_test, y_test))


if __name__ == "__main__":
    main()
