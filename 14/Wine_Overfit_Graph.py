from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import numpy
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# seed 값 설정 (동일한 랜덤값을 추출해주기 위해서 랜덤 시드값을 설정)
numpy.random.seed(3) 
tf.random.set_seed(3) 

# 데이터 입력
df_pre = pd.read_csv('./dataset/wine.csv', header=None) # 데이터 셋을 불러오도록 한다.
df = df_pre.sample(frac=0.15) # 전체 샘플 중 15%만 불러오도록 파라미터값을 준다.
dataset = df.values
X = dataset[:, 0:12] # 12개의 피쳐를 X값에 담는다.
Y = dataset[:, 12] # 마지막 class를 Y값에 담는다.

# 모델 설정
model = Sequential()
model.add(Dense(30, input_dim=12, activation="relu")) # 12개의 input과 30개의 output을 가진 입력층 생성
model.add(Dense(12, activation="relu")) # 12개의 input과 8개의 output을 가진 은닉층 생성
model.add(Dense(8, activation="relu")) # 8개의 input과 1개의 output을 가진 은닉층 생성
model.add(Dense(1, activation="sigmoid")) # 한 개의 input을 가지는 출력층 생성 (활성화 함수 : 시그모이드)

# 모델 컴파일
model.compile(loss="binary_crossentropy", # 오차 함수는 binary_corssentropy 함수를 사용.
        optimizer="adam", # 최적화 함수는 adam 함수를 사용
        metrics=["accuracy"]) # 결과값을 accuracy 형태로 보여지도록 함

# 모델 저장 폴더 설정
MODEL_DIR = './model/' # 모델이 저장될 폴더 경로 지정
if not os.path.exists(MODEL_DIR): # 해당 경로 파일이 존재하지 않는다면
    os.mkdir(MODEL_DIR) # 해당 폴더를 생성하라

# 모델 저장 조건 설정
modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5" # 모델명을 (에포크 수 - 오차).hdf5 로 저장

# 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_loss", 
    verbose=1, save_best_only=True) # 해당 함수의 진행 사항을 출력하도록 verbose 값을 1로 설정

# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=100)

model.fit(X, Y, validation_split=0.2, epochs=3500, batch_size=500,
    verbose=0, callbacks=[early_stopping_callback, checkpointer])

