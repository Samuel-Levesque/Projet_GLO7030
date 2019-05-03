
from torch.autograd import Variable


from data_set_file import create_huge_data_set,create_encoding_deconding_dict,generate_random_dataset,create_dict_nb_ligne
from model_creation import create_model
from trainning import  train_model,load_model_weights,create_scheduler


import torch

import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import random



def imshow(img_tensor,title):

    npimg = img_tensor.numpy()
    # print(npimg)
    plt.imshow(npimg,cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def get_history(path_start_from_existing_model):
    # Loading state
    checkpoint = torch.load(path_start_from_existing_model)

    history = checkpoint["history"]
    best_acc = checkpoint["best_acc"]

    print(best_acc)
    history.display_accuracy()



def simule_pioche(n_epoch,nsim=10000):

    percent_seen_list=[]
    percent_unique_list=[]

    for sim in range(nsim):
        all_seen=[]
        for epoch in range(n_epoch):
            all_data=list(range(0,150000))
            pick=random.sample(all_data,500)

            all_seen.extend(pick)

        percent_seen_list.append(len(np.unique(all_seen))/150000)
        percent_unique_list.append(len(np.unique(all_seen))/len(all_seen))

    print(np.mean(percent_seen_list))
    print(1-np.mean(percent_unique_list))



def show_mistakes(modele_name,path_data,use_gpu):
    def calcul_metric_concours(model, val_loader, use_gpu,dec_dict):
        model.train(False)
        true = []
        pred = []
        val_loss = []
        pred_top3 = []

        criterion = nn.CrossEntropyLoss()
        model.eval()

        for j, batch in enumerate(val_loader):

            inputs, targets = batch
            print(targets)
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            inputs = Variable(inputs, volatile=True)
            targets = Variable(targets, volatile=True)
            output = model(inputs)
            predictions = output.max(dim=1)[1]

            t1 = inputs[0, 0, :, :].cpu()
            label_true=dec_dict[targets.item()]
            label_pred=dec_dict[predictions.item()]


            imshow(t1,"True: {}, Pred: {}".format(label_true,label_pred))

            predictions_top_3 = output.topk(3)[1]






        model.train(True)







    enc_dict, dec_dict = create_encoding_deconding_dict(path_data)
    nb_ligne_dict = create_dict_nb_ligne(path_data)

    # Model
    model = create_model(use_gpu)

    if use_gpu:
        model.cuda()
    data_test = create_huge_data_set(path_data, nb_rows=100, size_image=224,
                                     skip_rows=range(1,100), encoding_dict=enc_dict)
    test_loader = DataLoader(data_test, batch_size=1,shuffle=True)

    model_final, history = load_model_weights(model, modele_name, type="best", use_gpu=use_gpu,
                                              get_history=True)


    calcul_metric_concours(model_final, test_loader, use_gpu,dec_dict)








if __name__ == "__main__":
    # get_history("poids/model_poids_general.tar")
    # get_history("poids/model_poids_mauvaise_classes_sampling.tar")
    get_history("poids/model_ensemble.tar")
    # simule_pioche(35,150)
    # show_mistakes("poids/model_poids_mauvaise_classes_sampling.tar",'D:/User/William/Documents/Devoir/Projet Deep/data/train_simplified/',True)