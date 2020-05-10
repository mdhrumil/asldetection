from PIL import Image

def getData(path):
    with open(path, "rb") as fp:
        image = Image.open(fp).convert('RGB')

    return image