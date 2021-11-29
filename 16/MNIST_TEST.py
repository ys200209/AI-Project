from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import sys
import cv2

# seed 값 설정
seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

# MNIST 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784).astype("float32") / 255
print("X_train : ", X_train)

X_test = X_test.reshape(X_test.shape[0], 784).astype("float32") / 255

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)


# 모델 프레임 설정
model = Sequential()
model.add(Dense(512, input_dim = 784, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(64, activation="relu")) # [32, 64, 128, 256, 512, ]
model.add(Dense(10, activation="softmax"))

modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", 
    verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)


# 모델 실행 환경 설정
model.compile(loss = "categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"])

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
        epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback, checkpointer])


# 테스트 정확도 출력
# print("\n Test Accuracy : %.4f" % (model.evaluate(X_test, Y_test)[1]))

'''
#CNN 모델 만들기
model = Sequential()
input_shape = (28, 28, 1)
model.add(Conv2D(512, kernel_size=(5, 5), strides=(1, 1), padding='same',
                 activation='relu',
                 input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()
'''

#CNN 모델 학습하기
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
 #       epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback, checkpointer])


'''
for x in test_num:
    for i in x:
        sys.stdout.write("%d\t" % i)
    sys.stdout.write("\n")
'''

# 이미지 불러와 출력

img = cv2.imread("C:\\Users\\Lee\\Desktop\\exam_22.png", cv2.IMREAD_GRAYSCALE)
# plt.imshow(gray)
# plt.show()

# 이미지 사이즈 변경
img = cv2.resize(255-img, (28, 28))
print("img.shape = ", img.shape)

test_num = img.flatten() / 255.0
test_num = test_num.reshape((-1, 28, 28, 1))
print("test_num.shape = ", test_num.shape)

# 이미지 숫자 테스트 
#print('The Answer is ', model.predict_classes(test_num))
print("X_test = ", X_test)
print("test_num = ", test_num)
predict_x=model.predict(test_num)
classes_x=np.argmax(predict_x,axis=1)
print("classes_x = ", classes_x)

'''
# 이미지 숫자 테스트 

# print("test_num", test_num)
y_pred = model.predict(test_num)
y_pred = np.round(y_pred).astype(int)

print('The Answer is ', y_pred)
'''
'''
for x in X_train[0]:
    for i in x:
        sys.stdout.write("%d\t" % i)
    sys.stdout.write("\n")

plt.imshow(X_train[0], cmap="Greys")
plt.show()
'''
