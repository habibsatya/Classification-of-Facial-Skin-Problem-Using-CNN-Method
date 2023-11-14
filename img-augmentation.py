from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img
import numpy as np
import os 

# image directory
# image_directory = 'dataset/raw/raw_bopeng/'
# image_directory = 'dataset/raw/raw_jerawat/'
image_directory = 'dataset/raw/raw_komedo/'

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect')

dataset = []
my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if (image_name.split('.')[1] == 'jpg'):
      img = load_img(image_directory + image_name)
      img = img_to_array(img)
      print(i, image_name, img.size)
      dataset.append(np.array(img))

x = np.array(dataset)

i = 0
for batch in datagen.flow(x, batch_size=50,
                          # save_to_dir='dataset/rgb/224/224_rgb_bopeng',
                          # save_to_dir='dataset/rgb/224/224_rgb_jerawat',
                          save_to_dir='dataset/rgb/224/224_rgb_komedo',
                          save_prefix='k',
                          save_format='jpg'):
    i += 1
    if i > 14:
        break

print(x.shape)
# for j in dataset:
#   print(np.shape(j))
# for count in range(1, 51, 1):
#     img = load_img('raw_dataset/raw_bopeng/bopeng{}.jpg'.format(count))
#     x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
#     x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

#     # the .flow() command below generates batches of randomly transformed images
#     # and saves the results to the `preview/` directory
#     i = 0
#     for batch in datagen.flow(x, batch_size=16,
#                               save_to_dir='D:/skincare-recommendation/image-model-2/jerawat', 
#                               save_prefix='j', 
#                               save_format='jpg'):
#         i += 1
#         if i == 8:
#             break  # otherwise the generator would loop indefinitely
