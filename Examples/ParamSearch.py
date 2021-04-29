import tensorflow as tf

from tensorflow.keras import datasets, layers, models, Sequential
import matplotlib.pyplot as plt
import math

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
print(len(train_images))


# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
##train_images, test_images = preProcess(train_images), preProcess(test_images)
##train_images, test_images = contrastCurve(train_images), contrastCurve(test_images)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

data_augmentation = Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
    input_shape=(32,32,3)),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.15),
    layers.experimental.preprocessing.RandomContrast(0.1),
  ]
)

def run_params(layer0,layer1,layer2,dense0,dense1):
    ## Creating the model
    model = models.Sequential()
    model.add(data_augmentation)

    model.add(layers.Conv2D(64,2,activation='relu',padding = "same", input_shape=(32, 32, 3)))

    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(128,3,activation = "relu",padding = "same"))
    model.add(layers.Dropout(0.25))

    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(256,4,activation = "relu",padding = "same"))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    ## Run the training
    history = model.fit(train_images, train_labels,batch_size=250, epochs=10,validation_data=(test_images, test_labels))
    return [layer0,layer1,layer2,dense0,dense1,history.history["accuracy"],history.history["val_accuracy"]]


from random import randint
runs = []

for i in range(0,100):
    f = open("ParamsHistory.txt",'a')
    layer0 = randint(16,512)
    layer1 = randint(16,512)
    layer2 = randint(16,512)
    dense0 = randint(128,1024)
    dense1 = randint(64,512)
    print("Running:\nLayer0:\t"+str(layer0)+"\nLayer1:\t"+str(layer1)+"\nLayer2:\t"+str(layer2)+"\nDense0:\t"+str(dense0)+"\nDense1:\t"+str(dense1))
    result = run_params(layer0,layer1,layer2,dense0,dense1)
    runs.append(result)
    f.write(str(result)+'\n')
    f.close()

    