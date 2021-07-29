import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout

main_path = "./landai/images"
image_names = []

for i in os.listdir(main_path):
  image_names.append(os.path.join(main_path,i))
  
image_names.sort()

mask_path = "./landai/masks"
mask_names = []

for i in os.listdir(mask_path):
  mask_names.append(os.path.join(mask_path,i))
  
mask_names.sort()

images=[]

for i in image_names:
  images.append(cv2.imread(i,1))

images = np.array(images)

masks=[]

for i in mask_names:
  masks.append(cv2.imread(i,1))

masks = np.array(masks)

masks = masks[:,:,:,:1]

x_train = images[:120]
x_test = images[120:]
y_train = masks[:120]
y_test = masks[120:]

y_train_1hot = tf.keras.utils.to_categorical(y_train)
y_test_1hot = tf.keras.utils.to_categorical(y_test)

plt.imshow(x_test[10,:,:,:].astype('uint8'))
plt.show()

plt.imshow(y_test[10,:,:,0].astype('uint8'))
plt.show()

x_in = Input(shape=(512, 512, 3))

x_temp = Conv2D(32, (3, 3), activation='relu', padding='same')(x_in)
x_temp = Dropout(0.25)(x_temp)
x_skip1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = MaxPooling2D((2,2))(x_skip1)
x_temp = Conv2D(32, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = Dropout(0.25)(x_temp)
x_skip2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = MaxPooling2D((2,2))(x_skip2)
x_temp = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = Dropout(0.25)(x_temp)
x_skip3 = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = MaxPooling2D((2,2))(x_skip3)
x_temp = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)
x_temp = Dropout(0.5)(x_temp)
x_temp = Conv2D(64, (3, 3), activation='relu', padding='same')(x_temp)

x_temp = Conv2DTranspose(64, (3, 3), activation='relu',  padding='same')(x_temp)
x_temp = Dropout(0.5)(x_temp)
x_temp = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu',  padding='same')(x_temp)
x_temp = Concatenate()([x_temp, x_skip3])
x_temp = Conv2DTranspose(64, (3, 3), activation='relu',  padding='same')(x_temp)
x_temp = Dropout(0.5)(x_temp)
x_temp = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu',  padding='same')(x_temp)
x_temp = Concatenate()([x_temp, x_skip2])
x_temp = Conv2DTranspose(32, (3, 3), activation='relu',  padding='same')(x_temp)
x_temp = Dropout(0.5)(x_temp)
x_temp = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu',  padding='same')(x_temp)
x_temp = Concatenate()([x_temp, x_skip1])
x_temp = Conv2DTranspose(32, (3, 3), activation='relu',  padding='same')(x_temp)
x_temp = Dropout(0.5)(x_temp)
x_temp = Conv2DTranspose(32, (3, 3), activation='relu',  padding='same')(x_temp)

x_temp = Conv2D(32, (1, 1), activation='relu', padding='same')(x_temp)
x_temp = Conv2D(32, (1, 1), activation='relu', padding='same')(x_temp)
x_out = Conv2D(5, (1, 1), activation='softmax', padding='same')(x_temp)

model = Model(inputs=x_in, outputs=x_out)

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()

"""## Modelin eğitilmesi"""

history = model.fit(x_train, y_train_1hot, validation_data=(x_test, y_test_1hot), epochs=10, batch_size=10, verbose=0)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

preds = model.predict(x_test)

preds = np.argmax(preds, axis=-1)

"""## Test setinin çıktıları"""

plt.imshow(preds[22, :, :])
plt.show()
plt.imshow(y_test[22, :, :, 0])
plt.show()
plt.imshow(x_test[22,:,:,:].astype('uint8'))
plt.show()

plt.imshow(preds[21, :, :])
plt.show()
plt.imshow(y_test[21, :, :, 0])
plt.show()
plt.imshow(x_test[21,:,:,:].astype('uint8'))
plt.show()

plt.imshow(preds[19, :, :])
plt.show()
plt.imshow(y_test[19, :, :, 0])
plt.show()
plt.imshow(x_test[19,:,:,:].astype('uint8'))
plt.show()

"""## Modele asıl istenen görsellerin verilmesi"""

img = cv2.imread("1.jpg",1)
img = img.reshape(1,*img.shape)
pred = model.predict(img)

plt.imshow(pred[0, :, :,0])
plt.show()
plt.imshow(img[0,:,:,:].astype('uint8'))
plt.show()

img = cv2.imread("2.jpg",1)
img = img.reshape(1,*img.shape)
pred = model.predict(img)

plt.imshow(pred[0, :, :,0])
plt.show()
plt.imshow(img[0,:,:,:].astype('uint8'))
plt.show()

results = model.evaluate(x_test, y_test_1hot, batch_size=10)
print("test loss, test acc:", results)

results = model.evaluate(x_train, y_train_1hot, batch_size=10)
print("test loss, test acc:", results)

model.save("first-model.h5")