import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

src = cv2.imread("./data/test/11.jpg", cv2.IMREAD_COLOR)
print(type(src))
print(src.shape)

dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.imshow("src", src)
cv2.imshow("dst", dst)

src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

fig = plt.figure()
ax1 = fig.add_subplot(1,5,1)
ax2 = fig.add_subplot(1,5,2)
ax3 = fig.add_subplot(1,5,3)
ax4 = fig.add_subplot(1,5,4)
ax5 = fig.add_subplot(1,5,5)
ax1.imshow(src[:,:,0])
ax2.imshow(src[:,:,1])
ax3.imshow(src[:,:,2])
ax4.imshow(src)
ax5.imshow(dst)


### save
cv2.imwrite("./test_output/gray_from_cv2.jpg", dst)
fig.savefig("./test_output/gray_from_fig.jpg")
Image.fromarray(dst).save("./test_output/gray_from_Image.jpg")

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()