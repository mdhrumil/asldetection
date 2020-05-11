import imageReader as ir
import numpy as np
from os import listdir
from os.path import isfile, join
import pickle as pkl
from random import shuffle
import time
import bz2

PATH = "C:\\Users\\mehta\\Desktop\\deeplearning\\projects\\asl\\data\\asl_alphabet_train\\"
dumpPATH = "C:\\Users\\mehta\\Desktop\\deeplearning\\projects\\asl\\binaries\\"
BATCH_SIZE = 1024

def loadData():
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    classes = ["A","B","C","D","del","E","F","G","H","I","J","K","L","M","N","nothing","O","P","Q","R","S","space","T","U","V","W","X","Y","Z"]

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

    return train_images, train_labels, test_images, test_labels


def createTrainBatches(train_images, train_labels):
    i = 0
    train_batches = len(train_images)//BATCH_SIZE

    for i in range(train_batches):
        train_images_shuffled = []
        train_labels_shuffled = []

        images_batch = train_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        labels_batch = train_labels[i*BATCH_SIZE : (i+1)*BATCH_SIZE]

        shuffleIndexTrain = list(range(len(images_batch)))
        shuffle(shuffleIndexTrain)

        for index in shuffleIndexTrain:
            train_images_shuffled.append(images_batch[index])
            train_labels_shuffled.append(labels_batch[index])

        image_array = np.array(train_images_shuffled)
        labels_array = np.array(train_labels_shuffled)

        with bz2.BZ2File(dumpPATH + "trainData_compressed_{}".format(i+1), "wb") as f:
            pkl.dump([image_array, labels_array], f)

    images_batch = train_images[BATCH_SIZE*train_batches:]
    labels_batch = train_labels[BATCH_SIZE*train_batches:]

    shuffleIndexTrain = list(range(len(images_batch)))
    shuffle(shuffleIndexTrain)

    for index in shuffleIndexTrain:
        train_images_shuffled.append(images_batch[index])
        train_labels_shuffled.append(labels_batch[index])

    image_array = np.array(train_images_shuffled)
    labels_array = np.array(train_labels_shuffled)

    with bz2.BZ2File(dumpPATH + "trainData_compressed_{}".format(train_batches + 1), "wb") as f:
        pkl.dump([image_array, labels_array], f)


def createTestBatches(test_images, test_labels):
    i = 0
    test_batches = len(test_images)//BATCH_SIZE

    for i in range(test_batches):
        test_images_shuffled = []
        test_labels_shuffled = []

        images_batch = test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        labels_batch = test_labels[i*BATCH_SIZE : (i+1)*BATCH_SIZE]

        shuffleIndexTest = list(range(len(images_batch)))
        shuffle(shuffleIndexTest)

        for index in shuffleIndexTest:
            test_images_shuffled.append(images_batch[index])
            test_labels_shuffled.append(labels_batch[index])

        image_array = np.array(test_images_shuffled)
        labels_array = np.array(test_labels_shuffled)

        with bz2.BZ2File(dumpPATH + "testData_compressed_{}".format(i+1), "wb") as f:
            pkl.dump([image_array, labels_array], f)

    images_batch = test_images[BATCH_SIZE*test_batches:]
    labels_batch = test_labels[BATCH_SIZE*test_batches:]

    shuffleIndexTest = list(range(len(images_batch)))
    shuffle(shuffleIndexTest)

    for index in shuffleIndexTest:
        test_images_shuffled.append(images_batch[index])
        test_labels_shuffled.append(labels_batch[index])

    image_array = np.array(test_images_shuffled)
    labels_array = np.array(test_labels_shuffled)

    with bz2.BZ2File(dumpPATH + "testData_compressed_{}".format(test_batches + 1), "wb") as f:
        pkl.dump([image_array, labels_array], f)


def main():

    start = time.time()

    train_images, train_labels, test_images, test_labels = loadData()

    createTrainBatches(train_images, train_labels)
    print("{} training data batches created.".format((len(train_images)//BATCH_SIZE) + 1))

    createTestBatches(test_images, test_labels)
    print("{} testing data batches created.".format((len(test_images)//BATCH_SIZE) + 1))

    end = time.time()

    print("Training and testing binaries written successfully in {} seconds.".format((end - start)))



if __name__ == '__main__':

    main()