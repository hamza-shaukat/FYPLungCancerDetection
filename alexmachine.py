import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,Input,Add
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import shutil
from tkinter import filedialog
from tkinter import messagebox

from tensorflow.keras import optimizers
import glob
import numpy as np
import sys
from tensorflow.keras.callbacks import Callback, CSVLogger,ModelCheckpoint;
#from livelossplot import PlotLossesKeras
from time import time
import json
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report


import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,Input,Add
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import shutil
from tkinter import filedialog
from tkinter import messagebox

from tensorflow.keras import optimizers
import glob
import numpy as np
import sys
from tensorflow.keras.callbacks import Callback, CSVLogger,ModelCheckpoint;
#from livelossplot import PlotLossesKeras
from time import time
import json
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
#np.random.seed(1000)
folder_path = 'C:/Users/Hamza/Desktop/dataset50'

folder_images = glob(folder_path + '/*/*.jpg')


#hog = cv2.HOGDescriptor()
'''from skimage.feature import hog
image_features = []
label_features=[]
total_images=len(folder_images)
for i,image_path in enumerate(folder_images):
    ir_=os.path.basename(os.path.dirname(image_path))
    image = cv2.imread(image_path)
    image1=image[...,2]
    #imagea=np.expand_dims(image1,-1)
    fd, hog_image = hog(image1, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=False)
    
    #plt.imshow(hog_image,cmap='gray')
    #break
    #features = hog.compute(image)
    image_features.append(fd)
    label_features.append(ir_)
    print(i+1, '/' , total_images,'-->',round((i+1)/total_images*100,4),'%')
X=np.array(image_features)
y=np.array(label_features)
np.save('C:/Users/Hamza/Desktop/saved data/xt.npy', X)
np.save('C:/Users/Hamza/Desktop/saved data/yt.npy', y)'''
X = np.load('C:/Users/Hamza/Desktop/saved data/xt.npy')
y = np.load('C:/Users/Hamza/Desktop/saved data/yt.npy')
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)






#np.random.seed(1000)
model_name="abc"
epoch=10
NUM_GPU = 1
img_width,img_height=256,256
path =filedialog.askdirectory(title='Please select a directory')

train_path=os.path.join(path,'train')# path+"/train"
test_path=path+"/valid"
classes=os.listdir(train_path)
print(classes)
if(K.image_data_format()=='channels_first'):
    input_shape=(3,img_width,img_height)
else:
    input_shape=(img_width,img_height,3)

   
train_datagen = ImageDataGenerator(rescale=1./255)
''',
     rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
'''
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 

train_batchsize = 16
val_batchsize = 16
 
train_generator = train_datagen.flow_from_directory(
        train_path,
        #color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=train_batchsize,
        class_mode='binary')
 
validation_generator = validation_datagen.flow_from_directory(
        test_path,
        #color_mode='grayscale',
        target_size=(img_width, img_height),
        batch_size=val_batchsize,
        class_mode='binary',
        shuffle=True)

'''i = Input((img_width, img_width, 3))

b1=Conv2D(filters=96, activation=("relu"), kernel_size=(11,11), strides=(4,4), padding="same")(i)
b1=MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")(b1)


b2=Conv2D(filters=384, kernel_size=(11,11),activation=("relu"), strides=(1,1), padding="same")(b1)
b2=MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")(b2)

b3=Conv2D(filters=256,activation=("relu"), kernel_size=(3,3), strides=(1,1), padding="same")(b2)

b4=MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")(b3)

#///////////////////////////////////////////////////////////////////////////////////////////////////////
c1=Conv2D(filters=96, activation=("relu"), kernel_size=(11,11), strides=(4,4), padding="valid")(i)



c2=Conv2D(filters=384, kernel_size=(11,11),activation=("relu"), strides=(1,1), padding="valid")(c1)


c3=Conv2D(filters=384,activation=("relu"), kernel_size=(3,3), strides=(1,1), padding="valid")(c2)
c3=Conv2D(filters=256,activation=("relu"), kernel_size=(8,8), strides=(7,7), padding="valid")(c2)


oo=c3

d1=Flatten()(oo)
d1=Dense(9216,activation=("relu"))(d1)
d1=Dropout(0.4)(d1)




d2=Dense(4096,activation=("relu"))(d1)
d2=Dropout(0.4)(d2)

d3=Dense(4096,activation=("relu"))(d2)


d4=Dense(1000,activation=("relu"))(d3)

d5=Dense(5,activation=("softmax"))(d4)

model = Model(inputs=i, outputs=d5, name=model_name)
model.summary()

#tf.keras.optimizers.SGD(learning_rate=0.1)
sgd = tf.keras.optimizers.SGD(learning_rate=0.01,decay=5.0E-4, momentum=0.9)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=["accuracy"])

keras.utils.plot_model(model, to_file="B.png", show_shapes=True)'''



model = Sequential()

model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding="valid"))#kernel_size=(11,11), strides=(4,4)
model.add(Activation("relu"))


model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))


model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding="same"))# kernel_size=(11,11), strides=(1,1)
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same"))
model.add(Activation("relu"))

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same"))
model.add(Activation("relu"))


model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same"))
model.add(Activation("relu"))


model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))


model.add(Flatten())
model.add(Dense(9216))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(4096))
model.add(Activation("relu"))
model.add(Dense(1000))

model.add(Activation("relu"))


model.add(Dense(len(classes)-1))
model.add(Activation("sigmoid"))

model.summary()
sgd = optimizers.Adam(lr=0.0001)

model.compile(loss=keras.losses.binary_crossentropy, optimizer=sgd, metrics=["binary_accuracy"])

#plot_model(model, show_shapes=True, to_file='B.png')
'''print("Feature extraction from the model")
feature_extractor = tensorflow.keras.Model(
   inputs=model.inputs,
   outputs=model.get_layer(name="OOu").output,
)'''
print(NUM_GPU)
if NUM_GPU != 1:
    model = keras.utils.multi_gpu_model(model, gpus=NUM_GPU)
checkpoint = ModelCheckpoint(os.path.join(path, 'Alexnet__CNN.h5'), monitor='val_accuracy', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False)


history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=epoch,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1, callbacks=[checkpoint])

'''history = model.fit_generator(
      (train_generator,validation_generator),
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=epoch,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1, callbacks=[checkpoint])'''

model.save_weights(os.path.join(path, "WeightAlexnetWeightCNN"))
#model.save(os.path.join(path, "AlexnetCNN.h5"))
print(history.history)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epoch)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



Y_pred = model.predict_generator(validation_generator,validation_generator.samples // validation_generator.batch_size+1)

y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
cm=confusion_matrix(validation_generator.classes, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

disp.plot(cmap=plt.cm.Blues)
plt.show()
print('Classification Report')

print(classification_report(validation_generator.classes, y_pred, target_names=classes))
