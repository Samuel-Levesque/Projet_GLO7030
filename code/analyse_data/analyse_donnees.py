import os
import random
import pandas as pd
import numpy as np
import ast
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader
from torch.utils.data import Dataset, DataLoader, ConcatDataset


path = 'D:/User/William/Documents/Devoir/Projet Deep/data/train_simplified/'
filenames = os.listdir(path)
filenames = [path + x for x in filenames]



class DoodlesDataset_mod(Dataset):
    """Doodles csv dataset.
    adapté de https://www.kaggle.com/leighplt/pytorch-starter-kit/notebook


    Dataset Pytorch pour une seul catégorie. Pour faire un dataset complet on concatène plusieurs de ces dataset

    """

    def __init__(self, csv_file, root_dir,nrows=1000,encoding_dict=None, mode='train', skiprows=None, size=256, transform=None):
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






        if self.mode == 'train':

            self.txt_label= csv_file[:-4]
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

        print(raw_strokes)

        # if self.transform:
        #     sample = self.transform(sample)
        if self.mode == 'train':
            return (sample[None] / 255).astype('float32'), self.label
        else:
            return (sample[None] / 255).astype('float32')



def draw_specific_vecteur(raw_strokes):
    def draw(raw_strokes, size=256, largeur_trait=6):
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

    sample = draw(raw_strokes, size=256, largeur_trait=6)

    print(raw_strokes)

    img=sample/255

    plt.imshow(img, cmap="gray")
    # for traits in raw_strokes:
    #     vecteur_x=traits[0]
    #     vecteur_y=traits[1]
    #     plt.plot(vecteur_x,vecteur_y,"ro")

    # plt.plot([73,24],[32,48],"ro")
    # plt.title("House")

    plt.axis("off")
    plt.show()


def show_nb_class_nb_data(filenames):


    total_draw=0
    for file_name in filenames:
        n = sum(1 for line in open(file_name)) - 1
        total_draw+=n


    print("Nombre de données: {}".format(total_draw)) #51 551 820
    print("Nombre de classe: {}".format(len(filenames))) #340


def imshow(img_tensor,title):

    npimg = img_tensor.numpy()
    # print(npimg)
    plt.imshow(npimg,cmap="gray")
    plt.title(title)
    plt.show()


def show_random_draw(path,class_name=None,how_many=100,how_many_per_class=1):

    filenames = os.listdir(path)
    filenames = [path + x for x in filenames]
    size_image = 256

    if class_name==None:
        for i in range(how_many):
            number=random.randint(0,len(filenames))
            csv_file = filenames[number].split('/')[-1]
            data_set=DoodlesDataset_mod(csv_file, path, size=size_image, mode="test")

            loader = DataLoader(data_set, batch_size=how_many_per_class, shuffle=True)

            i=0
            for image in loader:
                i+=1
                t1 = image[0, 0, :, :]
                imshow(t1, title=csv_file[:-4])
                if i>=how_many_per_class:
                    break
            pass



    else:
        csv_file=class_name+".csv"
        data_set = DoodlesDataset_mod(csv_file, path, size=size_image, mode="test")

        loader=DataLoader(data_set,batch_size=how_many_per_class,shuffle=True)

        for image in loader:

            t1=image[0,0,:,:]
            imshow(t1,title=csv_file[:-4])



if __name__ == "__main__":


    # show_random_draw(path,class_name="house")
    # show_random_draw(path,how_many_per_class=1)


    vecteur=[[[0, 73, 83, 97, 150, 152, 170, 205, 255, 239, 203, 207, 245], [114, 0, 72, 128, 37, 30, 126, 82, 31, 72, 142, 145, 142]]]
    vecteur=[[[37, 50, 120, 0, 27, 63, 97, 133, 134, 130, 64, 18, 45, 151, 167, 168, 145, 55, 59, 79, 137, 147, 142], [3, 0, 0, 77, 73, 76, 82, 94, 97, 100, 125, 155, 158, 160, 163, 168, 182, 221, 225, 228, 244, 251, 255]]]
    vecteur=[[[95, 80, 70, 70, 77, 77], [0, 41, 83, 127, 161, 184]], [[120, 121, 115, 115], [45, 94, 138, 171]], [[171, 168], [62, 112]], [[167, 180, 180, 168], [62, 100, 119, 161]], [[255, 248, 240, 226], [27, 53, 117, 142]], [[0, 32, 79, 141, 178, 228, 218, 208, 200, 178, 151, 86, 61, 45, 43, 51, 51], [33, 33, 41, 45, 45, 35, 140, 162, 167, 166, 159, 149, 148, 143, 115, 82, 22]], [[140, 137, 139, 144, 155, 159, 158, 142], [72, 74, 94, 96, 94, 86, 76, 70]], [[145, 145, 133, 149, 161, 155], [97, 131, 144, 130, 145, 131]], [[128, 158], [101, 107]]]
    #2 frog:
    # vecteur=[[[72, 76, 105, 146, 193, 201, 227, 240, 251, 254, 255, 248], [57, 49, 34, 32, 35, 31, 4, 0, 1, 6, 33, 44]], [[69, 58, 47, 38, 27, 23, 23, 47, 50, 38, 34, 16, 2, 1, 12, 16, 50, 41, 37, 48, 57, 73, 79, 78, 85, 93, 103, 105, 109, 120, 129, 139, 156], [46, 32, 26, 28, 40, 51, 59, 72, 95, 108, 124, 151, 184, 197, 199, 197, 131, 161, 194, 196, 188, 143, 180, 199, 205, 204, 190, 159, 148, 209, 212, 201, 152]], [[248, 223, 209, 245, 251, 254, 247, 234, 222, 213, 212, 222, 220, 215, 193, 190, 191, 184, 174, 168, 162, 154, 151, 147, 144, 146, 149], [44, 75, 99, 145, 157, 180, 196, 198, 194, 184, 175, 191, 209, 211, 204, 195, 187, 209, 214, 185, 197, 202, 202, 196, 169, 159, 158]], [[52, 76, 144, 173, 200, 215, 230], [81, 86, 84, 78, 77, 72, 62]], [[240, 236, 243, 245, 236], [10, 26, 21, 16, 16]], [[53, 49], [45, 46]]]
    vecteur=[[[83, 61, 54, 50, 57, 53], [35, 35, 41, 55, 84, 84]], [[66, 50], [69, 72]], [[84, 83, 90, 83, 84, 93, 104], [69, 77, 87, 69, 64, 62, 64]], [[122, 130, 139, 133, 121, 117, 122, 130, 136], [88, 86, 78, 69, 67, 78, 87, 88, 79]], [[167, 154, 144, 141, 149, 159, 164, 162, 159, 154, 151, 144, 136, 130, 133, 168, 175], [73, 63, 66, 80, 89, 84, 73, 72, 80, 115, 122, 125, 122, 115, 110, 108, 102]], [[25, 33, 54, 114, 155, 176, 189, 201, 205, 204, 197, 172, 151, 133, 101, 62, 36, 15, 6, 0, 1, 7, 13, 27, 40, 57], [94, 103, 116, 138, 140, 137, 131, 114, 103, 91, 76, 50, 33, 25, 15, 13, 21, 37, 51, 74, 91, 105, 110, 114, 113, 106]], [[29, 22, 9], [117, 135, 152]], [[145, 141, 139], [139, 151, 177]], [[174, 179, 189, 202, 212, 226, 247, 253, 252, 237, 228, 215, 195], [53, 22, 8, 0, 0, 7, 33, 46, 57, 87, 93, 96, 89]], [[214, 233, 250, 251, 245, 238], [4, 18, 44, 77, 88, 91]], [[213, 196, 186, 179, 181, 189, 199, 214, 234, 249, 254, 246, 228, 207, 194, 175, 171, 171, 176, 195, 217, 229, 240, 254, 255, 244, 231, 208, 193, 188, 184], [14, 15, 20, 37, 61, 82, 95, 99, 94, 81, 64, 42, 19, 8, 8, 25, 38, 57, 72, 89, 95, 93, 87, 69, 52, 30, 16, 6, 11, 17, 30]], [[198, 198], [38, 38]], [[223, 223], [32, 33]], [[229, 205], [60, 67]], [[214, 220], [32, 23]], [[205, 195], [34, 30]]]

    #house
    # vecteur=[[[20, 93, 134, 142, 150, 192, 174, 68, 26, 0, 1, 21, 31, 40, 187, 223, 224, 209, 201, 199], [90, 29, 3, 0, 3, 91, 95, 86, 79, 72, 93, 196, 228, 233, 253, 255, 250, 203, 150, 102]], [[126, 126, 129, 195], [235, 175, 171, 169]]]

    draw_specific_vecteur(vecteur)