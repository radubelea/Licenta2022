import pandas as pd
import numpy as np
import itertools
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
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
    filename = 'model_KNN.sav'
    leaf_size = list(range(5, 10))
    n_neighbors = list(range(1, 5))
    p = [1, 2]
    params_knn = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    classifier = KNeighborsClassifier(n_neighbors=5)
    knn_grid = RandomizedSearchCV(estimator=classifier, param_distributions=params_knn, n_iter=20, cv=3, verbose=2,
                                  random_state=42, n_jobs=-1)
    knn_grid.fit(x_train, y_train)
    best_grid = knn_grid.best_estimator_
    pickle.dump(best_grid, open(filename, 'wb'))
    # classifier = pickle.load(open(filename, 'rb'))
    print("Loaded the model")
    y_pred = best_grid.predict(x_test)
    print("Predicted the labels")
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    acc = accuracy_score(y_test, y_pred)
    cvs = cross_val_score(classifier, x, y, cv=5)
    print('Accuracy: ', acc)
    print('Cross Val Score: ', cvs)
    cm = confusion_matrix(y_test, y_pred)
    create_confusion_matrix(cm, labels)


if __name__ == "__main__":
    main()
