import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


data=[]
labels=[]
sets=36
cur_path=os.getcwd()
print("Completed")

for i in range(sets):
    path=os.path.join(cur_path,"Training",str(i))
    images=os.listdir(path)
    print(str(i),end=" ")
    for file_name in images:
        try:
            image= Image.open(path+ "/" +file_name)
            image=image.resize((30,30))
            image= np.array(image)
            data.append(image)
            labels.append(i)
        
        except Exception as e:
            print("Error loading image:"+file_name+"\n"+e)
            break
            
                              
data=np.array(data)
labels=np.array(labels)

print("\n",data.shape, labels.shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=35)
y_test2=y_test
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
print(y_test[9])

y_train=to_categorical(y_train,36)
y_test=to_categorical(y_test,36)

print(y_test[9])
plt.imshow(x_test[9])

model=Sequential()
shape=x_train.shape[1:]
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=shape))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(36, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Compiled")
Compiled

history=model.fit(x_train, y_train, batch_size=32, epochs=3, validation_data=(x_test, y_test))
model.save("tsr.h5")

loaded=load_model('tsr.h5')

right=0
tot=0
for i in range(sets):
    data=[]
    path=os.path.join(cur_path,"Training",str(i))
    images=os.listdir(path)
    for img in images:
        image= Image.open(path+ "/" +img)
        image=image.resize((30,30))
        image= np.array(image)
        data.append(image)
    data=np.array(data)
    pred=loaded.predict_classes(data)
    for j in pred:
        tot+=1
        if(j==i):
            right+=1
print(right/tot)

plt.figure()
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(history.history['accuracy'],label='training accuracy')
plt.plot(history.history['val_accuracy'],label='val accuracy')
plt.legend()
