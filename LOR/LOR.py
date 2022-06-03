import pandas as pd
import numpy as np
import itertools
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from utils.data import retrieve_data
import matplotlib.pyplot as plt
import pickle


def create_confusion_matrix(cm, labels):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def main():
    x, y, x_train, x_test, y_train, y_test = retrieve_data()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print("Split the data into test and train values")
    filename = 'model_LOR.sav'
    C = np.logspace(-4, 4, 10)
    penalty = ['l2']
    params_lor = dict(C=C, penalty=penalty)
    classifier = LogisticRegression(multi_class='ovr', max_iter=100000)
    # classifier.fit(x_train, y_train)
    lor_grid = GridSearchCV(estimator=classifier, param_grid=params_lor, cv=5, verbose=1, scoring='accuracy')
    lor_grid.fit(x_train, y_train)
    best_grid = lor_grid.best_estimator_
    pickle.dump(classifier, open(filename, 'wb'))
    print("Trained the model")
    # classifier = pickle.load(open(filename, 'rb'))
    print("Loaded the model")
    y_pred = best_grid.predict(x_test)
    print("Predicted the labels")
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

    acc = accuracy_score(y_test, y_pred)
    r2s = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print('Accuracy: ', acc)
    print('R2 Score: ', r2s)
    print('Mean Squared Error: ', mse)


if __name__ == "__main__":
    main()
