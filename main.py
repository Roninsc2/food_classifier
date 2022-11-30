import random
from imutils import paths
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.framework.config import list_physical_devices, set_memory_growth

from pic import *
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard
from resnet import *


# visualization train data
def visual(X_train, Y_train):
    index = np.random.choice(np.arange(len(X_train)), 24, replace=False)
    figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(16, 9))
    for item in zip(axes.ravel(), X_train[index], Y_train[index]):
        axes, image, target = item
        axes.imshow(image)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title(target)
    plt.show()


# visualizations prediction err
def visual_incorrect(Epochs, Hist):
    # From second epoch, because error on 0 and 1 are too big.
    N = np.arange(2, Epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, Hist.history["loss"][2:], label="train_loss")
    plt.plot(N, Hist.history["val_loss"][2:], label="val_loss")
    plt.plot(N, Hist.history["accuracy"][2:], label="train_acc")
    plt.plot(N, Hist.history["val_accuracy"][2:], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch №")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


# prediction error analysis
def incorrect(X_test, Y_test, pred):
    food = ('Bread', 'Dessert', 'Meat', 'Soup')
    incorrect_predictions = []
    for i, (p, e) in enumerate(zip(pred, Y_test)):
        predicted, expected = np.argmax(p), np.argmax(e)
        if predicted != expected:
            incorrect_predictions.append((i, X_test[i], predicted, expected))
    print(len(incorrect_predictions))
    figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(16, 12))
    for item in zip(axes.ravel(), incorrect_predictions):
        axes, inc_pred = item
        axes.imshow(inc_pred[1])
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title(f'p: {food[inc_pred[2]]}; e: {food[inc_pred[3]]}')
    plt.show()
    confusion = tf.math.confusion_matrix(Y_test.argmax(axis=1), pred.argmax(axis=1))
    print(confusion)


# test data prepare
imagePaths = list(paths.list_images('Food'))
random.seed(256)
random.shuffle(imagePaths)
input_width = 128
sp = Preprocessor(input_width, input_width)
dsl = DatasetLoader(preprocessors=[sp])
(data, labels) = dsl.load(imagePaths)
data = data.astype('float32') / 255
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, random_state=256)

print('Train:',
      np.count_nonzero(trainY == 'Meat') / trainY.size,
      np.count_nonzero(trainY == 'Soup') / trainY.size,
      np.count_nonzero(trainY == 'Bread') / trainY.size,
      np.count_nonzero(trainY == 'Dessert') / trainY.size)
print('Test:',
      np.count_nonzero(testY == 'Meat') / testY.size,
      np.count_nonzero(testY == 'Soup') / testY.size,
      np.count_nonzero(testY == 'Bread') / testY.size,
      np.count_nonzero(testY == 'Dessert') / testY.size)

visual(trainX, trainY)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10.0,
                                                          width_shift_range=0.10,
                                                          height_shift_range=0.10,
                                                          shear_range=0.10, zoom_range=0.10,
                                                          horizontal_flip=True,
                                                          vertical_flip=True)

# nn create
physical_devices = list_physical_devices('GPU')
set_memory_growth(physical_devices[0], True)
lr_reducer = ReduceLROnPlateau(factor=0.1,
                               cooldown=0,
                               patience=8,
                               min_lr=0.5e-6,
                               verbose=1)
early_stopper = EarlyStopping(min_delta=0.0005,
                              patience=10,
                              verbose=1)
csv_logger = CSVLogger('./resnet_food.csv')
tb = TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)

model = ResnetBuilder.build_resnet((3, input_width, input_width), 4, 34)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

# train nn
epochs = 100
batch_size = 64
H = model.fit(datagen.flow(trainX, trainY),
              epochs=epochs,
              validation_data=(testX, testY),
              verbose=1,
              callbacks=[lr_reducer, early_stopper, csv_logger, tb])
model.save_weights('./food_weights.hdf5')

# nn analysis
results = model.evaluate(testX, testY)
print(results)
predictions = model.predict(testX)
for index, probability in enumerate(predictions[0]):
    print(f'{index}:{probability:.10%}')
incorrect(testX, testY, predictions)
visual_incorrect(79, H)
