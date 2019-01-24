import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


# -> Load Data

train_path = 'train'
test_path = 'test'
valid_path = 'valid'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['dog', 'cat'], batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['dog', 'cat'], batch_size=8)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['dog', 'cat'], batch_size=4)

# -> Build and train CNN

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224,224,3)),
    Flatten(),
    Dense(2, activation='softmax'),
])

model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=5, validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)

# -> Predict using model
test_imgs, test_labels = next(test_batches)
test_labels = test_labels[:,0]
print(test_labels)
predictions = model.predict_generator(test_batches, steps=1, verbose=0)
print(predictions)

# -> Confusion Matrix


cm = confusion_matrix(test_labels, predictions[:,0])

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[1, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

cm_plot_labels = ['cat', 'dog']
plt.figure()
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
plt.show()

# from random import randint
# from sklearn.preprocessing import MinMaxScaler



# # -> Start creating sample data

# train_labels = []
# train_samples = []

# for i in range(50):
#     random_younger = randint(13, 64)
#     train_samples.append(random_younger)
#     train_labels.append(1)

#     random_older = randint(65, 100)
#     train_samples.append(random_older)
#     train_labels.append(0)

# for i in range(1000):
#     random_younger = randint(13, 64)
#     train_samples.append(random_younger)
#     train_labels.append(0)

#     random_older = randint(65, 100)
#     train_samples.append(random_older)
#     train_labels.append(1)

# # -> Preprocessing

# train_labels = np.array(train_labels)
# train_samples = np.array(train_samples)

# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1, 1))

# # -> End creating sample data

# model = Sequential([
#     Dense(16, input_shape=(1,), activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(2, activation='softmax')
# ])

# # -> Alternate Syntax
# # model = Sequential()
# # model.add(Dense(5, input_shape=(3,)))
# # model.add(Activation('relu'))

# model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # model.loss='sparse_categorical_crossentropy'
# # model.loss

# model.fit(scaled_train_samples, train_labels, validation_split=0.1, batch_size=10, epochs=20, shuffle=True, verbose=2)

# # -> Create test data
# test_samples = []
# test_lables = []

# for i in range(10):
#     random_younger = randint(13, 64)
#     test_samples.append(random_younger)
#     test_lables.append(1)

#     random_older = randint(65, 100)
#     test_samples.append(random_older)
#     test_lables.append(0)

# for i in range(200):
#     random_younger = randint(13, 64)
#     test_samples.append(random_younger)
#     test_lables.append(0)

#     random_older = randint(65, 100)
#     test_samples.append(random_older)
#     test_lables.append(1)

# test_lables = np.array(train_labels)
# test_samples = np.array(train_samples)

# scaled_test_samples = scaler.fit_transform((test_samples).reshape(-1,1))

# predictions = model.predict(scaled_test_samples, batch_size=10, verbose=2)
# for i in predictions:
#     print(i)

# rounded_predictions = model.predict_classes(scaled_test_samples, batch_size=10, verbose=2)
# for i in rounded_predictions:
#     print(i)




# # -> Saving a model
# model.save('super_basic_model.h5')

# # -> Loading a model
# from keras.models import load_model
# new_model = load_model('super_basic_model.h5')
# new_model.summary()
# print(new_model.get_weights())
# print(new_model.optimizer)

# # -> Save as JSON (Save only architecture)
# json_string = model.to_json()
# print(json_string)

# # -> Model reconstruction from JSON
# from keras.models import model_from_json
# model_architecture = model_from_json(json_string)
# model_architecture.summary()

# # -> Save weights only
# model.save_weights('super_basic_model_weights.h5')

# model2 = Sequential([
#     Dense(16, input_shape=(1,), activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(2, activation='softmax')
# ])

# model2.load_weights('super_basic_model_weights.h5')