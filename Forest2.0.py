# Result: accuracy ~30%
# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class_names = ['Pop_Rock', 'Electronic', 'Rap', 'Jazz', 'Latin', 'RnB',
               'International', 'Country', 'Reggae', 'Blues']


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

    print('Analyzing training labels:')
    for i in range(1, 11, 1):
        print(f'Label {i}({class_names[i - 1]}): {len(np.where(train_labels_accuracy.values == i)[0])} ({round((len(np.where(train_labels_accuracy.values == i)[0]) / len(train_labels_accuracy)) * 100, 2)}%)')

    # Read the values
    train_labels_accuracy = train_labels_accuracy.values
    train_data_accuracy = train_data_accuracy.values
    test_data_accuracy = test_data_accuracy.values

    # Scale the data before feeding them in the Neural Network
    scaler = StandardScaler()
    train_data_resized = scaler.fit_transform(train_data_accuracy)
    test_data_resized = scaler.transform(test_data_accuracy)

    classifier = RandomForestClassifier(n_estimators=1000, criterion='gini', random_state=42, verbose=1)
    classifier.fit(train_data_resized, np.ravel(train_labels_accuracy))

    print(classifier.feature_importances_)

    # Outputs the predictions: array of 6544 (0 to 6543)
    predictions = classifier.predict(test_data_resized)
    print(f'Total predictions: {len(predictions)}')

    for j in range(1, 11, 1):
        index_acc = np.where(predictions == j)
        print(f'{class_names[j - 1]}: {index_acc[0].size} ({round((index_acc[0].size / len(predictions)) * 100, 2)}%)')
    print(f'Last value: {predictions[len(predictions) - 1]}')
    id_num = range(1, len(predictions) + 1)
    df = pd.DataFrame({'Sample_id': id_num, 'Sample_label': predictions}, columns=['Sample_id', 'Sample_label'])
    df.to_csv('treeforest.csv')


if __name__ == "__main__":
    main()
