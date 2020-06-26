from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
# Training
train_dir = 'C:/Users/Jeanette/Desktop/cursoP/Red_Neuronal_Convolucional_rnc/Piedra_papel_tijera/training_set'
# train_dir = 'C:/Users/Jeanette/Desktop/cursoP/Red_Neuronal_Convolucional_rnc/dataset-colposcopy2/training_set'
# train_cin1_dir = os.path.join(train_dir,'CIN1')
# train_cin2_dir = os.path.join(train_dir,'CIN2')
# train_cin3_dir = os.path.join(train_dir,'CIN3')


train_cin1_dir = os.path.join(train_dir,'paper')
train_cin2_dir = os.path.join(train_dir,'rock')
train_cin3_dir = os.path.join(train_dir,'scissors')
# Validation
vali_dir = 'C:/Users/Jeanette/Desktop/cursoP/Red_Neuronal_Convolucional_rnc/Piedra_papel_tijera/validation_set'
# vali_dir = 'C:/Users/Jeanette/Desktop/cursoP/Red_Neuronal_Convolucional_rnc/dataset/validation_set'
vali_cin1_dir = os.path.join(vali_dir,'paper')
vali_cin2_dir = os.path.join(vali_dir,'rock')
vali_cin3_dir = os.path.join(vali_dir,'scissors')

test_dir = 'C:/Users/Jeanette/Desktop/cursoP/Red_Neuronal_Convolucional_rnc/Piedra_papel_tijera/test_set'

print('total training cin1 images:', len(os.listdir(train_cin1_dir)))
print('total training cin2 images:', len(os.listdir(train_cin2_dir)))
print('total training cin3 images:', len(os.listdir(train_cin3_dir)))

print('total validation cin1 images:', len(os.listdir(vali_cin1_dir)))
print('total validation cin2 images:', len(os.listdir(vali_cin2_dir)))
print('total validation cin3 images:', len(os.listdir(vali_cin3_dir)))

train_cin1_files = os.listdir(train_cin1_dir)
# print(train_cin1_files[:5])
train_cin2_files = os.listdir(train_cin2_dir)
# print(train_cin2_files[:5])
train_cin3_files = os.listdir(train_cin3_dir)
# print(train_cin3_files[:5])
# -----visualizar las imágenes ------
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# pic_index = 2
# next_cin1 = [os.path.join(train_cin1_dir, fname) 
#     for fname in train_cin1_files[pic_index-2:pic_index]]

# next_cin2 = [os.path.join(train_cin2_dir, fname) 
#     for fname in train_cin2_files[pic_index-2:pic_index]]

# next_cin3 = [os.path.join(train_cin3_dir, fname) 
#     for fname in train_cin3_files[pic_index-2:pic_index]]

# print("p:::::: ",next_cin1,"---",next_cin2,"---",next_cin3)
# for i, img_path in enumerate(next_cin1 + next_cin2 + next_cin3):
#  print(img_path)
#  img = mpimg.imread(img_path)
#  plt.imshow(img)
#  plt.axis('Off')
#  plt.show()
# ---------------------------Preprocessing------------------------
training_datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2, # recorre la imagen horizontalmente una razon alaatoria 
                height_shift_range=0.2, # recorre la imagen verticalmente una razon alaatoria
                rescale=1./255,
                shear_range=0.2, # Intensidad de corte (ángulo de corte en sentido antihorario en grados)
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Toma la ruta a un directorio y genera lotes de datos aumentados
train_generator = training_datagen.flow_from_directory(
                train_dir,
                batch_size= 64,         # 32 (default)
                target_size=(150,150),  # (256,256)--cargar todas las imágenes a un tamaño específico
                class_mode='categorical' # tipos de clasificación
)
validation_generator = validation_datagen.flow_from_directory(
                vali_dir,
                batch_size=64,        
                target_size=(150,150),
                class_mode='categorical'
)
test_generator = validation_datagen.flow_from_directory(
                test_dir,
                batch_size=64,        
                target_size=(150,150),
                class_mode='categorical'
)
batch_size = 64
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size
print (steps_per_epoch)      # 1
print (validation_steps)     # 0


# Define el modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)), # RGB colores
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.B

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
#    tf.keras.layers.Dense(1, activation='sigmoid') # Salida:cat-dog
    tf.keras.layers.Dense(3, activation='softmax') 
])
model.summary()
# Compile
model.compile(
        loss = 'categorical_crossentropy', optimizer='adam',
        metrics=['acc']
        )
# Train network
history = model.fit(
        train_generator,
        steps_per_epoch = steps_per_epoch,     # cuántos lotes de imágenes definen una sola época
        epochs=5,
        validation_data = validation_generator, 
        validation_steps = validation_steps,   # cuántos lotes en el conjunto de los datos de validación define una época
        verbose=1   # 0 = silent, 1 = progress bar, 2 = one line
        )
# Guardar el modelo con Keras model.save()
model_path = 'C:/Users/Jeanette/Desktop/cursoP/Red_Neuronal_Convolucional_rnc/Cancer_cervical-web/model/modelo2.h5'
model.save(model_path)
# ---------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
history_dict = history.history
print(history_dict.keys())

acc      = history.history[     'acc' ]
val_acc  = history.history[ 'val_acc' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(1,len(acc)+1,1)

plt.plot ( epochs,     acc, 'r--', label='Training acc'  )
plt.plot ( epochs, val_acc,  'b', label='Validation acc')
plt.title ('Training and validation accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')

plt.legend()
plt.figure()

test_lost, test_acc= model.evaluate(test_generator)  
print ("Test Accuracy:", test_acc)

# Correr  acá
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(
#    'C:/Users/Jeanette/Desktop/cursoPython/Red_Neuronal_Convolucional_rnc/dataset-colposcopy/validation_set/CIN1/12.jpg',
   'C:/Users/Jeanette/Desktop/cursoP/Red_Neuronal_Convolucional_rnc/Piedra_papel_tijera/test_set/rock/rock7.png',
   target_size=(150,150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
resultado = model.predict(test_image, batch_size=10)
#
#if resultado[0][0] == 0:
#    prediccion = 'gato'
#else:
#    prediccion = 'perro'
#print(resultado,'--->', prediccion)    
   