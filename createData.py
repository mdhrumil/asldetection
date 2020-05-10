import imageReader as ir
import numpy as np
from os import listdir
from os.path import isfile, join
import pickle as pkl
from random import shuffle
import time

PATH = "C:\\Users\\mehta\\Desktop\\deeplearning\\projects\\asl\\data\\asl_alphabet_train\\"
dumpPATH = "C:\\Users\\mehta\\Desktop\\deeplearning\\projects\\asl\\binaries\\"


def main():

    train_images = []
    train_labels = []
    train_images_shuffled = []
    train_labels_shuffled = []
    test_images = []
    test_labels = []
    test_images_shuffled = []
    test_labels_shuffled = []

    classes = ["A","B","C","D","del","E","F","G","H","I","J","K","L","M","N","nothing","O","P","Q","R","S","space","T","U","V","W","X","Y","Z"]

    start = time.time()


    for label in classes:
        filesPath = PATH + label
        files = [f for f in listdir(filesPath) if isfile(join(filesPath, f))]
        temp = 0

        for file in files:
            imagePath = filesPath + "\\" + file
            image = ir.getData(imagePath)
            array = np.array(image)
            temp += 1
            if(temp > 2400):
                test_images.append(array)
                test_labels.append(label)
            else:
                train_images.append(array)
                train_labels.append(label)

        print("images for class: {} imported!".format(label))

    
    shuffleIndexTrain = list(range(len(train_images)))
    shuffleIndexTest = list(range(len(test_images)))

    shuffle(shuffleIndexTrain)
    shuffle(shuffleIndexTest)

    for i in shuffleIndexTrain:
        train_images_shuffled.append(train_images[i])
        train_labels_shuffled.append(train_labels[i])

    for i in shuffleIndexTest:
        test_images_shuffled.append(test_images[i])
        test_labels_shuffled.append(test_labels[i])


    train_data = np.array(train_images_shuffled)
    test_data = np.array(test_images_shuffled)

    with open(dumpPATH + "trainData", "wb") as f:
        pkl.dump([train_data, train_labels_shuffled], f, protocol = 4)

    with open(dumpPATH + "testData", "wb") as f:
        pkl.dump([test_data, test_labels_shuffled], f, protocol = 4)

    end = time.time()

    print("Training and test binaries written successfully in {} seconds.".format((end - start)))



if __name__ == '__main__':

    main()