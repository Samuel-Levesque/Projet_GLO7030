import os

path = 'D:/User/William/Documents/Devoir/Projet Deep/data/train_simplified/'

filenames = os.listdir(path)
filenames = [path + x for x in filenames]


file=filenames[0]
total_draw=0
for file_name in filenames:
    n = sum(1 for line in open(file)) - 1
    total_draw+=n


print("Nombre de données: {}".format(total_draw)) #51 551 820
print("Nombre de classe: {}".format(len(filenames))) #340
