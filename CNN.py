#Bryan Jonathan Melo De Oliveira
import tensorflow as tf
import keras as K

# Imports
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

print("Versão do TensorFlow:", tf.__version__)
print("Versão do Keras:", K.__version__)

#Inicializando a Rede Neural Convolucional
classifier = Sequential()

#Passo 1 - Primeira Camada de Convolucao
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#Passo 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adicionando a Segunda Camada de Convolucao
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D( pool_size = (2, 2)))

#Passo 3 - Flattening
classifier.add(Flatten())

#Passo 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compilando a rede
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Criando os objetos train_datagen e validation_datagen com as regras de pre-processamento das imagens
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

#Pre-processamento das imagens de treino e validacao
training_set = train_datagen.flow_from_directory('dataset_treino', #training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_set = validation_datagen.flow_from_directory('dataset_validation',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')

#Executando o treinamento
classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 5, validation_data = validation_set, validation_steps = 2000)


#Primeira Imagem 
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset_teste/2216.jpg', target_size = (64, 64))
test_image - image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Cachorro'
else:
    prediction = 'Gato'

Image(filename = 'dataset_teste/2216.jpg')

#tive um problema, nao consegui encontrar como resolver online
#FileNotFoundError: [WinError 3] O sistema não pode encontrar o caminho especificado: 'dataset_treino'