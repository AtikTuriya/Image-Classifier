from .data_utils import TEST_TRANSFORM, MEAN, STD
import PIL
import numpy as np
import matplotlib.pyplot as plt


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with PIL.Image.open(image_path) as image:
        image = TEST_TRANSFORM(image).numpy()
    
    return image

def imshow(image, ax=None, title=None):
    """imshow for tensor"""
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.transpose((1, 2, 0))
    std   = np.array(STD)
    mean  = np.array(MEAN)
    image = std * image + mean

    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


if __name__ == '__main__':
    from sys import argv
    from sys import exit as sysexit
    if len(argv) < 2:
        print("You must provide a path to an image")
        sysexit()
    image_path = argv[1]
    imshow(process_image(image_path))
    plt.show()
