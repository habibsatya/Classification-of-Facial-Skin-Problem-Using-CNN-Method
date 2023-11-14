import cv2 as cv
import glob

def resize(img, ratio):
    width, height = (ratio, ratio)
    dim = (width, height)

    # resize image
    img_resize = cv.resize(img, dim)
    
    return img_resize

# image path
# image_path = 'dataset/rgb/224/224_rgb_bopeng/*.jpg'
# image_path = 'dataset/rgb/224/224_rgb_jerawat/*.jpg'
# image_path = 'dataset/rgb/224/224_rgb_komedo/*.jpg'
# image_path = 'dataset/grayscale/224/224_gray_bopeng/*.jpg'
# image_path = 'dataset/grayscale/224/224_gray_jerawat/*.jpg'
image_path = 'dataset/grayscale/224/224_gray_komedo/*.jpg'

# target path
# target_path = 'dataset/rgb/128/128_rgb_bopeng/'
# target_path = 'dataset/rgb/128/128_rgb_jerawat/'
# target_path = 'dataset/rgb/128/128_rgb_komedo/'
# target_path = 'dataset/grayscale/128/128_gray_bopeng/'
# target_path = 'dataset/grayscale/128/128_gray_jerawat/'
target_path = 'dataset/grayscale/128/128_gray_komedo/'

my_images = list(glob.glob(image_path))
new_size = 128

for i, image_name in enumerate(my_images):
    image = cv.imread(image_name)
    image_resize = resize(image, new_size)
    cv.imwrite(target_path + 'k_gray_128_{}.jpg'.format(i), image_resize)