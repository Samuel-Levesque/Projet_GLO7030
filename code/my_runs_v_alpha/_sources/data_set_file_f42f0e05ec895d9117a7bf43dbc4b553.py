

import  random
import warnings
warnings.filterwarnings('ignore') # to suppress some matplotlib deprecation warnings

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import ast
import cv2

import matplotlib.pyplot as plt


import os

from torch.utils.data import Dataset, DataLoader, ConcatDataset





#Pour une classe
class DoodlesDataset(Dataset):
    """Doodles csv dataset.
    adapté de https://www.kaggle.com/leighplt/pytorch-starter-kit/notebook


    """

    def __init__(self, csv_file, root_dir, mode='train', nrows=1000, skiprows=None, size=224, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. ex :airplane.csv
            root_dir (string): Directory with all the images.
            mode (string): Train or test mode.
            nrows (int): Number of rows of file to read. Useful for reading pieces of large files.
            skiprows (list-like or integer or callable):
                    Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file.
            size (int): Size of output image.
            transform (callable, optional): Optional transform to be applied  (pas utile pour l'instant)
                on a sample.
        """
        self.root_dir = root_dir
        file = os.path.join(self.root_dir, csv_file)
        self.size = size
        self.mode = mode

        #ESSAI
        random_row=False
        if random_row:
            n = sum(1 for line in open(file)) - 1  # number of records in file (excludes header)

            skip = sorted(random.sample(range(1, n + 1), n - nrows))
            self.doodle = pd.read_csv(file, usecols=['drawing'], skiprows=skip,nrows=nrows)  # Data set pandas

        else:

            self.doodle = pd.read_csv(file, usecols=['drawing'], nrows=nrows, skiprows=skiprows) #Data set pandas


        # self.transform = transform



        if self.mode == 'train':
            self.label = csv_file.replace(' ', '_')[:-4]

    @staticmethod
    def _draw(raw_strokes, size=256, largeur_trait=6):
        BASE_SIZE = 256

        img = np.full((BASE_SIZE, BASE_SIZE), 255,dtype=np.uint8)
        for t, stroke in enumerate(raw_strokes):
            for i in range(len(stroke[0]) - 1):
                color = 0

                _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                             (stroke[0][i + 1], stroke[1][i + 1]), color, largeur_trait)

        if size != BASE_SIZE:
            return cv2.resize(img, (size, size))
        else:
            return img


    def __len__(self):
        return len(self.doodle)

    def __getitem__(self, idx):

        raw_strokes = ast.literal_eval(self.doodle.drawing[idx])
        sample = self._draw(raw_strokes, size=self.size, largeur_trait=6)

        # if self.transform:
        #     sample = self.transform(sample)
        if self.mode == 'train':
            return (sample[None] / 255).astype('float32'), self.label
        else:
            return (sample[None] / 255).astype('float32')


#Pour toutes les classe, nb_row par classe

def create_huge_data_set(path,nb_rows,size_image,filenames=None):
    '''
    On pourra modifer plus tard

    :param path:  path où se trouve le dossier avec les csv
    :param nb_rows:  Nombre de rows par classes
    :param size_image:
    :param filenames: si on veut clase particulière ex : [airplane.csv, angel.csv]
    :return:
    '''


    if filenames==None:
        filenames = os.listdir(path)



    doodles = ConcatDataset([DoodlesDataset(fn,path,nrows=nb_rows, size=size_image) for fn in filenames])

    return doodles


def imshow(img_tensor):

    npimg = img_tensor.numpy()
    # print(npimg)
    plt.imshow(npimg,cmap="gray")
    plt.show()


if __name__ == "__main__":

    path = 'D:/User/William/Documents/Devoir/Projet Deep/data/mini_train/'
    # path = 'D:/User/William/Documents/Devoir/Projet Deep/data/train_simplified/'

    filenames = os.listdir(path)
    filenames = [path + x for x in filenames]

    size_image = 224

    select_nrows = 1000


    csv_file=filenames[0].split('/')[-1]

    #Créer data set pour un csv file en particulier
    # essai=DoodlesDataset(csv_file, path,nrows=select_nrows, size=size_image)






    # loader=DataLoader(essai,batch_size=10)
    # for image, label in loader:
    #     print(image)
    #     t1=image[0,0,:,:]
    #     #imshow(t1)
    #     print(label)


    doodles = ConcatDataset([DoodlesDataset(fn.split('/')[-1], path,
                                               nrows=select_nrows, size=size_image) for fn in filenames])

    loader = DataLoader(doodles, batch_size=2,shuffle=True)

    i=0
    for image, label in loader:
        # print(image)
        t1 = image[0, 0, :, :]
        t2=image[1,0,:,:]
        # imshow(t1)
        # imshow(t2)
        i+=2
        print(i)
        print(label)

    print("end")