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

# -> Fine Tune Model

vgg16_model = keras.applications.vgg16.VGG16()

type(vgg16_model)

vgg16_model.summary()

vgg16_model.layers.pop()
vgg16_model.outputs = [vgg16_model.layers[-1].output]
vgg16_model.layers[-1].outbound_nodes = []

model = Sequential()

for layer in vgg16_model.layers:
    model.add(layer)

model.summary()

for layer in model.layers:
    layer.trainable = False

model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=5, validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)

# -> Predict using model
test_imgs, test_labels = next(test_batches)
test_labels = test_labels[:,0]
print(test_labels)
predictions = model.predict_generator(test_batches, steps=1, verbose=0)
print(predictions)

# -> Confusion Matrix

cm = confusion_matrix(test_labels, np.round(predictions[:,0]))

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