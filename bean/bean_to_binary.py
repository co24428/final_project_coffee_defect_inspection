import cv2
import glob
import os
import sys

# output_path = './data/normal_binary_data/'
# output_path = './data/broken_binary_data/'
# output_path = './data/normal_rotated_binary_data/'
output_path = './data/broken_rotated_binary_data/'

# imgs = glob.glob(os.path.join("./data/normal_sec_processed_data",'*.jpg'))
# imgs = glob.glob(os.path.join("./data/broken_sec_processed_data",'*.jpg'))
# imgs = glob.glob(os.path.join("./data/normal_rotated_data",'*.jpg'))
imgs = glob.glob(os.path.join("./data/broken_rotated_data",'*.jpg'))


x = 0
for img in imgs:
    src = cv2.imread(img , cv2.IMREAD_COLOR)
    # cv2.imshow("src", src)

    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    ret, dst = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    dst = cv2.bitwise_not(dst)
    # cv2.imshow("dst", dst)

    
    cv2.imwrite(output_path + "rotated_binary_bean_" + str(x) + ".jpg", dst)
    x += 1
    print(x)
# cv2.waitKey(0)
# cv2.destroyAllWindows()