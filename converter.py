from cv2 import *
import numpy as np
import matplotlib.pyplot as plt

class converter:
    def __init__(self):
      #read the 'image.png' and converting it to GrayScale Image for feature extraction
      self.img=imread('MNIST_IMAGE.png',0)


    def convert(self):
      #Resizing The image to img 28X28 pixels
      self.img=resize(self.img,(28,28),interpolation=INTER_AREA)
      #imshow("text",img)
      #waitKey()##

    def pixel_break_down(self):
      #laying out all the features(pixels) in a single row
       self.x_test=[]
       print(self.img.shape)
       for i in range(28):
         for j in range(28):
              data=self.img[i][j]
              self.x_test.append(data)
       self.x_test1=np.asarray(self.x_test)

    def thresholding(self):
        for index, val in enumerate(self.x_test1):
            if val == 255:
                self.x_test1[index] = 0
            elif val > 0:
                self.x_test1[index] = 255

    def load(self):
        plt.imshow(self.x_test1.reshape(28,28))


if __name__ == '__main__':
    ob=converter()
    ob.convert()
    ob.pixel_break_down()
    ob.load()

