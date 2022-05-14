from keras.preprocessing.image import ImageDataGenerator

data_dir = 'C:/Users/chess/valid/flat' #Due to the structure of ImageDataGenerator, you need to have another folder under train contains your data, for example: data/train/faces
save_dir = 'C:/Users/chess/valid/flat/a2'

datagen = ImageDataGenerator(rescale=1./255)

resized = datagen.flow_from_directory(data_dir, target_size=(224, 224),
                                save_to_dir=save_dir,
                                color_mode="rgb", # Choose color mode
                                class_mode='categorical',
                                shuffle=True,
                                save_prefix='N',
                                save_format='jpg', # Formate
                                batch_size=1)
for i in range(len(resized)):
    resized.next()

data_dir = 'C:/Users/chess/valid/inthestreet' #Due to the structure of ImageDataGenerator, you need to have another folder under train contains your data, for example: data/train/faces
save_dir = 'C:/Users/chess/valid/inthestreet/a1'

datagen = ImageDataGenerator(rescale=1./255)
resized = datagen.flow_from_directory(data_dir, target_size=(224, 224),
                                save_to_dir=save_dir,
                                color_mode="rgb", # Choose color mode
                                class_mode='categorical',
                                shuffle=True,
                                save_prefix='N',
                                save_format='jpg', # Formate
                                batch_size=1)

for i in range(len(resized)):
    resized.next()
