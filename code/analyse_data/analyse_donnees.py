import os
import random
from data_set_file import DoodlesDataset,create_huge_data_set
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader

path = 'D:/User/William/Documents/Devoir/Projet Deep/data/train_simplified/'
filenames = os.listdir(path)
filenames = [path + x for x in filenames]


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
    size_image = 224

    if class_name==None:
        for i in range(how_many):
            number=random.randint(0,len(filenames))
            csv_file = filenames[number].split('/')[-1]
            data_set=DoodlesDataset(csv_file, path, size=size_image, mode="test")

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
        data_set = DoodlesDataset(csv_file, path, size=size_image, mode="test")

        loader=DataLoader(data_set,batch_size=how_many_per_class,shuffle=True)

        for image in loader:

            t1=image[0,0,:,:]
            imshow(t1,title=csv_file[:-4])






show_random_draw(path,class_name="face")
# show_random_draw(path,how_many_per_class=2)