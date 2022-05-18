import pandas as pd
import numpy as np
import itertools
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import Input
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from utils.data import retrieve_data
import matplotlib.pyplot as plt
import pickle


def create_model():
    model = Sequential()
    model.add(Input(shape=(60,)))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['Accuracy', 'Precision', 'Recall'],
                  loss_weights=None,
                  weighted_metrics=None,
                  run_eagerly=None,
                  steps_per_execution=None
                  )
    return model


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
    x_train, x_test, y_train, y_test = retrieve_data()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print("Split the data into test and train values")
    filename = 'model_FFNN.sav'
    classifier = create_model()
    classifier.fit(x_train,
                   y_train,
                   batch_size=16,
                   epochs=50,
                   verbose='auto',
                   callbacks=None,
                   shuffle=True,
                   sample_weight=None,
                   initial_epoch=0,
                   steps_per_epoch=len(x_train) / 16,
                   validation_steps=None,
                   validation_batch_size=None,
                   max_queue_size=10,
                   workers=1,
                   use_multiprocessing=False,
                   )
    pickle.dump(classifier, open(filename, 'wb'))
    # classifier = pickle.load(open(filename, 'rb'))
    print("Loaded the model")
    y_pred = classifier.predict(x_test)
    print("Predicted the labels")
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

    acc = accuracy_score(y_test, y_pred)
    r2s = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print('Accuracy: ', acc)
    print('R2 Score: ', r2s)
    print('Mean Squared Error: ', mse)
    cm = confusion_matrix(y_test, y_pred)
    create_confusion_matrix(cm, labels)


if __name__ == "__main__":
    main()
