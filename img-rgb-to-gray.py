import cv2 as cv
import glob

# image path
# image_path = 'dataset/rgb/224/224_rgb_bopeng/*.jpg'
# image_path = 'dataset/rgb/224/224_rgb_jerawat/*.jpg'
image_path = 'dataset/rgb/224/224_rgb_komedo/*.jpg'

# target directory
# target_directory = 'dataset/grayscale/224/224_gray_bopeng/'
# target_directory = 'dataset/grayscale/224/224_gray_jerawat/'
target_directory = 'dataset/grayscale/224/224_gray_komedo/'

my_images = list(glob.glob(image_path))

for i, image_name in enumerate(my_images):
    image = cv.imread(image_name)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite(target_directory + 'k_gray_{}.jpg'.format(i), image_gray)

# cv.imwrite('b.jpg', image_gray)
# cv.imshow('gray', image_gray)
# cv.waitKey(0)
