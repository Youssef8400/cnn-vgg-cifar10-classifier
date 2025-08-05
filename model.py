from keras._tf_keras.keras.datasets import cifar10
from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.layers import Conv2D  , MaxPooling2D , Dense , Flatten , Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train_raw = y_train.copy()
y_test_raw = y_test.copy()

x_train = x_train.astype('float64') / 255.0
x_test = x_test.astype('float64') / 255.0


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

import pandas as pd


# Afficher les dimensions : 

print("Dimension de x_train : " ,x_train.shape)
print("Dimension de x_test : " , x_test.shape)
print("Dimension de y_train : " , y_train.shape)
print("Dimension de y_test : " , y_test.shape)


# Nombre de classe :

classes =np.unique(y_train)
print(classes)
num = len(classes)
print("Nombre des classes :" , num)

# Nombre d'exemple pour chaque classe : 
unique, counts = np.unique(y_train_raw, return_counts=True)
print("Nombre d'exemples par classe :")
for cls, count in zip(unique, counts):
    print(f"Classe {cls} : {count} exemples")

model = Sequential()


# bloc 1 :
model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Dropout(0.2))  

# bloc 2 :
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Dropout(0.3))  

# bloc 3 :
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Dropout(0.4))  

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))  
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test))

train_scores = model.evaluate(x_train, y_train)
print("Model evaluate (X_train):")
print(f"{model.metrics_names[0]} : {train_scores[0]*100:.2f} %")
print(f"{model.metrics_names[1]} : {train_scores[1]*100:.2f} %")

test_scores = model.evaluate(x_test, y_test)
print("Model evaluate (X_test):")
print(f"{model.metrics_names[0]} : {test_scores[0]*100:.2f} %")
print(f"{model.metrics_names[1]} : {test_scores[1]*100:.2f} %")


y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  
y_true_classes = np.argmax(y_test, axis=1)  

conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='plasma')
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion")
plt.show()

model.save("youssef999.h5")

