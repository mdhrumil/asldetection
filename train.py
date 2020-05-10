#import tensorflow as tf
import imageReader as ir
import numpy as np


PATH = "C:\\Users\\mehta\\Desktop\\deeplearning\\projects\\asl\\data\\asl_alphabet_train\\A\\A1.jpg"


def main():
    image = ir.getData(PATH)
    array = np.array(image)

    print(array.shape)
    
    


if __name__ == '__main__':

    main()