import os
import re
import random
from PIL import Image
import numpy as np
import config


NEG = 0
POS = 1
IMAGE_PIXELS = config.IMAGE_PIXELS
VGG_MEAN = config.VGG_MEAN
CATEGORIES = config.CATEGORIES

# manager puzzle images, generate batches for training and testing.
class PuzzleSet(object):

    train_set = None
    test_set = None

    # return (images, labels) for training
    # the real size may be smaller because some images may fail to load.
    def train_next_batch(self, size):
        batch = random.sample(self.train_set, size)
        return self.load_batch(batch)

    # return (images, labels) for testing
    def validation_batch(self):
        return self.load_batch(self.test_set)

    def load_batch(self, batch):
        images = []
        labels = []
        for sample in batch:
            path, cat = sample
            try:
                image = load_standard_img(path)
                images.append(image)
                labels.append(cat)
            except Exception:
                continue
        return np.concatenate(images, axis=0), build_np_labels(labels)


# generate numpy label 
def build_np_labels(labels):
    size = len(labels)
    res = np.zeros([size, 2], dtype="float32")
    for i in range(size):
        res[i][labels[i]] = 1
    return res


# load image from path and transform into numpy format
def load_standard_img(path):
    image = Image.open(path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((IMAGE_PIXELS, IMAGE_PIXELS))
    x = np.array(image, dtype="float32")

    def imagenet_preprocess(x):
        x = x[:, :, ::-1]
        x[:, :, 0] -= VGG_MEAN[0]
        x[:, :, 1] -= VGG_MEAN[1]
        x[:, :, 2] -= VGG_MEAN[2]
        return x

    return np.expand_dims(imagenet_preprocess(x), axis=0)


# get an instance of PuzzleSet with train_set and test_set already setted.
def read_data_sets(basepath):
    puzzleset = PuzzleSet()
    
    train_neg_dir = os.path.join(basepath, 'train/neg/')
    train_pos_dir = os.path.join(basepath, 'train/pos/')
    test_neg_dir = os.path.join(basepath, 'test/neg/')
    test_pos_dir = os.path.join(basepath, 'test/pos/')

    train_set = []
    test_set = []

    train_neg_filepaths = [
        train_neg_dir + f for f in os.listdir(train_neg_dir) if re.search(r'(\.jpg|\.jpeg|\.png)$', f)
    ]
    train_pos_filepaths = [
        train_pos_dir + f for f in os.listdir(train_pos_dir) if re.search(r'(\.jpg|\.jpeg|\.png)$', f)
    ]
    for path in train_neg_filepaths:
        train_set.append((path, NEG))
    for path in train_pos_filepaths:
        train_set.append((path, POS))
    random.shuffle(train_set)

    test_neg_filepaths = [
        test_neg_dir + f for f in os.listdir(test_neg_dir) if re.search(r'(\.jpg|\.jpeg|\.png)$', f)
    ]
    test_pos_filepaths = [
        test_pos_dir + f for f in os.listdir(test_pos_dir) if re.search(r'(\.jpg|\.jpeg|\.png)$', f)
    ]
    for path in test_neg_filepaths:
        test_set.append((path, NEG))
    for paht in test_pos_filepaths:
        test_set.append((path, POS))
    random.shuffle(test_set)

    puzzleset.train_set = train_set
    puzzleset.test_set = test_set
    return puzzleset
