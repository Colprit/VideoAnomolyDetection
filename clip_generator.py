from os import listdir
from os.path import join, isdir
import numpy as np
import keras
from PIL import Image


def get_IDs(dataset_path, clip_len, stride):
    IDs = []
    for folder in sorted(listdir(dataset_path)):
        directory_path = join(dataset_path, folder)
        if isdir(directory_path):
            img_files = [
                file
                for file in sorted(listdir(directory_path))
                if file.endswith('tif')
            ]
            # to reduce training set size
            img_files = img_files[:100]
            for i in range(len(img_files) - clip_len*stride):
                IDs.append((
                    folder, 
                    [ img_files[j] for j in range(i, i+clip_len*stride, stride) ]
                ))
    return IDs


class DataGenerator(keras.utils.Sequence):

    def __init__(self,
        dataset_path,
        clip_len,
        stride,
        dim, # (10, 256, 256)
        batch_size,
        n_channels, # 1
        shuffle
    ) -> None:
        self.dataset_path = dataset_path
        self.dim = dim 
        self.batch_size = batch_size 
        self.n_channels = n_channels
        self.shuffle = shuffle

        # list(tuple)
        # (folder, list(image files)) for one clip
        self.IDs = []
        for s in range(stride):
            self.IDs.extend(get_IDs(dataset_path, clip_len, s+1))
        
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        l = np.floor( len(self.IDs) / self.batch_size )
        return int(l)


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generates indexes of the batch
        start = self.batch_size*index
        indexes = self.indexes[start:start+self.batch_size]

        batch_IDs = [self.IDs[k] for k in indexes]

        X = self.__data_generation(batch_IDs)
        return (X, X)


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def __data_generation(self, batch_IDs):
        'Generates data containing batch_Size samples'

        # Initialization
        X = np.zeros(shape=(self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, (folder, img_files) in enumerate(batch_IDs):
            clip = []
            for img_file in img_files:
                img_path = join(self.dataset_path, folder, img_file)
                img = Image.open(img_path).resize((256, 256))
                img = np.array(img, dtype=np.float32) / 256.0
                clip.append(img)
            # Store sample
            X[i,:,:,:,:]  = np.array(clip).reshape(*self.dim, 1)

        return X
