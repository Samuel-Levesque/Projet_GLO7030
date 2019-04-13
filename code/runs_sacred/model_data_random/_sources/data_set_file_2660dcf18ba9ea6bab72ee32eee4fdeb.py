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
from utility import  save_object,load_object


def create_encoding_deconding_dict(path_data):
    '''
    Crée un dictionnaire d'encoding des labels et un dictionnaire de decoding des labels
    :param path_data:
    :return:
    '''
    filenames = os.listdir(path_data)
    filenames=sorted(filenames)

    en_dict = {}
    counter = 0
    for fn in filenames:
        en_dict[fn[:-4].split('/')[-1].replace(' ', '_')] = counter
        counter += 1

    dec_dict = {v: k for k, v in en_dict.items()}


    save_object(en_dict,"saves_obj/en_dict.pk")
    save_object(dec_dict, "saves_obj/dec_dict.pk")

    return en_dict,dec_dict

    pass




#Pour une classe
class DoodlesDataset(Dataset):
    """Doodles csv dataset.
    adapté de https://www.kaggle.com/leighplt/pytorch-starter-kit/notebook


    Dataset Pytorch pour une seul catégorie. Pour faire un dataset complet on concatène plusieurs de ces dataset

    """

    def __init__(self, csv_file, root_dir,nrows,encoding_dict=None, mode='train', skiprows=None, size=224, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. ex :airplane.csv
            root_dir (string): Directory with all the csv.
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



        self.doodle = pd.read_csv(file, usecols=['drawing'], nrows=nrows, skiprows=skiprows) #Data set pandas


        # self.transform = transform



        if self.mode == 'train':

            self.txt_label= csv_file.replace(' ', '_')[:-4]
            self.label = encoding_dict[self.txt_label]





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
def create_huge_data_set(path,nb_rows=1000,size_image=224,encoding_dict=None,skip_rows=None,filenames=None,mode="train"):
    '''
    Concatène les dataset de plusieurs classes

    :param path:  path où se trouve le dossier avec les csv
    :param nb_rows:  Nombre de rows par classes
    :param size_image:
    :param filenames: si on veut des classe particulières ex : [airplane.csv, angel.csv]
    :return:
    '''


    if filenames==None:
        filenames = os.listdir(path)



    doodles = ConcatDataset([DoodlesDataset(fn,path,nrows=nb_rows, size=size_image,
                                            skiprows=skip_rows,encoding_dict=encoding_dict,mode=mode)
                             for fn in filenames])

    return doodles




def generate_random_dataset( path, nb_row_valid,nb_rows_test,nb_rows,dict_nb_lignes, size_image=224, encoding_dict=None,filenames=None,
                             use_acc_proportionate_sampling=False):
    '''

    Pour chaque classe dans filenames, on prend nb_rows données aléatoire dans le fichier

    :param path:
    :param nb_row_valid:
    :param nb_rows_test:
    :param nb_rows:
    :param size_image:
    :param encoding_dict:
    :param filenames:
    :return:
    '''


    if filenames==None:
        filenames = os.listdir(path)




    if use_acc_proportionate_sampling:
        if os.path.isfile("saves_obj/dict_acc_per_class_valid.pk"):
            dict_acc_class=load_object("saves_obj/dict_acc_per_class_valid.pk")
        else:
            print("Aucun dictionnaire d'accuracy par classe trouvé; sampling uniforme utilisé")
            use_acc_proportionate_sampling=False





    nb_lignes_skip = nb_row_valid + nb_rows_test
    list_dataset=[]


    dict_nb_row_used_per_class={}

    for fn in filenames:
        n = dict_nb_lignes[fn]
        skip =list(range(1,nb_lignes_skip)) +sorted(random.sample(range(nb_lignes_skip,n), n - nb_rows-nb_lignes_skip))


        if use_acc_proportionate_sampling:
            acc=dict_acc_class[fn[:-4]]
            new_rows=round((1.1-acc)*nb_rows )

        else:
            new_rows=nb_rows
        dict_nb_row_used_per_class[fn]=new_rows

        data_set=DoodlesDataset(fn, path, nrows=new_rows, size=size_image,
                       skiprows=skip, encoding_dict=encoding_dict, mode="train")
        list_dataset.append(data_set)

    doodles = ConcatDataset(list_dataset)


    print("Nombre de données d'entraînement (total:{}):".format(sum(dict_nb_row_used_per_class.values())),dict_nb_row_used_per_class)

    return doodles



def create_dict_nb_ligne(path,filenames=None):
    '''
    dictionnaire du nombre de ligne dans les fichiers csv
    :param path:
    :return:
    '''

    if filenames==None:
        filenames = os.listdir(path)

    dict_nb_ligne={}

    for fn in filenames:

        print(fn)
        n = sum(1 for line in open(path + fn)) - 1
        dict_nb_ligne[fn]=n

    save_object(dict_nb_ligne,"saves_obj/dict_nb_ligne.pk")

    return dict_nb_ligne




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
    # essai=DoodlesDataset(csv_file, path,nrows=select_nrows, size=size_image,skiprows=range(1,10))






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