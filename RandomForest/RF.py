import pandas as pd
import numpy as np
import itertools
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from pprint import pprint
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
    filename = 'model_RF.sav'
    n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    pprint(random_grid)
    classifier = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=classifier, param_distributions=random_grid, n_iter=20, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(x_train, y_train)
    best_random = rf_random.best_estimator_
    pickle.dump(best_random, open(filename, 'wb'))
    # classifier = pickle.load(open(filename, 'rb'))
    print("Loaded the model")
    y_pred = best_random.predict(x_test)
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
