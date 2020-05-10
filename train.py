import numpy as np
import pickle as pkl
import numpy as np
import bz2

dataPATH = "C:\\Users\\mehta\\Desktop\\deeplearning\\projects\\asl\\binaries\\"

def main():
    trainDump = None
    testDump = None

    with bz2.BZ2File(dataPATH + "trainData_compressed_68", 'rb') as trainFile:
        trainDump = pkl.load(trainFile)

    with bz2.BZ2File(dataPATH + "testData_compressed_17", 'rb') as testFile:
        testDump = pkl.load(testFile)

    print("Train data matrix shape: {}".format(trainDump[0].shape))
    print("Train data labels shape: {}".format(len(trainDump[1])))

    print("Test data matrix shape: {}".format(testDump[0].shape))
    print("Test data labels shape: {}".format(len(testDump[1])))

if __name__ == '__main__':

    main()