from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import pandas as pd

COLUMN_NAMES = [
        'SepalLength', 
        'SepalWidth',
        'PetalLength', 
        'PetalWidth', 
        'Species'
        ]

training_dataset = pd.read_csv('iris_training.csv', names=COLUMN_NAMES, header=0)
train_x = training_dataset.iloc[:, 0:4].values
train_y = training_dataset.iloc[:, 4].values

encoding_train_y = np_utils.to_categorical(train_y)

test_dataset = pd.read_csv('iris_test.csv', names=COLUMN_NAMES, header=0)
test_x = test_dataset.iloc[:, 0:4].values
test_y = test_dataset.iloc[:, 4].values

encoding_test_y = np_utils.to_categorical(test_y)

model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(train_x, encoding_train_y, epochs=300, batch_size=10)

scores = model.evaluate(test_x, encoding_test_y)
print("\nAccuracy: %.2f%%" % (scores[1]*100))