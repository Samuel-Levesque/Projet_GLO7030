import torch.optim as optim
import torch.nn as nn
from deeplib.training import validate
from deeplib.datasets import train_valid_loaders
from deeplib.history import History
import torch
import copy
from torch.optim.lr_scheduler import  LambdaLR
import random
import numpy as np
import os.path
from utility import  save_object

from test_metrics import calcul_metric_concours


class LRPolicy(object):
    '''
    pourrait recevoir d'autres arguments
    '''
    def __init__(self, start_lr,rate=0):
        self.rate = rate
        self.start_lr=start_lr
    def __call__(self, epoch):

        # return self.start_lr*np.exp(-self.rate*epoch) +0.0009
        return self.start_lr



def create_optimizer():
    pass



def create_scheduler(start_lr,optimizer,type="constant"):
    if type=="constant":
        schedul=LRPolicy(start_lr=start_lr)




    return LambdaLR(optimizer, lr_lambda=schedul)





def load_model_weights(model,path_weights,type="best",use_gpu=False,get_history=False):
    model_structure= copy.deepcopy(model)

    if use_gpu:
        checkpoint = torch.load(path_weights)

    else:
        checkpoint = torch.load(path_weights, map_location="cpu")

    if type=="best":
        model_weights = checkpoint["best_model_weights"]
    elif type=="last":
        model_weights = checkpoint["model_state_dict"]

    model_structure.load_state_dict(model_weights)

    if get_history:
        history = checkpoint["history"]
        return model_structure,history

    else:
        return model_structure






def train_model(model, train_loader,val_loader, n_epoch,scheduler,optimizer,criterion, use_gpu=False,
                path_save=None,path_start_from_existing_model=None,val_acc_class_save_name=None):




    if path_start_from_existing_model is not None and os.path.isfile(path_start_from_existing_model):

        # Loading state
        checkpoint = torch.load(path_start_from_existing_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        next_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        history = checkpoint["history"]
        best_acc = checkpoint["best_acc"]
        best_model_weights = checkpoint["best_model_weights"]
        scheduler.load_state_dict(checkpoint["lr_scheduler_state"])

        print("Modèle chargé pour entraînement")

    else:
        # best_model_weights = copy.deepcopy(model.state_dict())
        history = History()
        next_epoch = 0
        best_acc=0
        print("Aucun modèle chargé pour entraînement")





    # Entrainement
    for epoch in range(0, n_epoch):
        model.train()
        scheduler.step()
        for j, batch in enumerate(train_loader):

            inputs, targets = batch
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            output = model(inputs)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

        train_acc, train_loss, train_top3_score, train_conf_mat, train_acc_per_class=calcul_metric_concours(model,train_loader,use_gpu,show_acc_per_class=True)
        val_acc, val_loss, val_top3_score, val_conf_mat, val_acc_per_class = calcul_metric_concours(model,val_loader,use_gpu,show_acc_per_class=True)

        #Current LR
        for param_group in optimizer.param_groups:
            current_lr=param_group["lr"]



        history.save(train_acc, val_acc, train_loss, val_loss, current_lr)
        print('Epoch {} -Train acc: {:.2f} -Val acc: {:.2f} -Train loss: {:.4f} - Val loss: {:.4f} -Train score top3 : {:.4f} -Val score top3 : {:.4f}'.format(epoch,
                                                                                                              train_acc,
                                                                                                              val_acc,
                                                                                                              train_loss,
                                                                                                              val_loss,train_top3_score,val_top3_score))
        #Accuracy par classe
        print(val_acc_per_class)
        if val_acc_class_save_name is not None:
            save_object(val_acc_per_class,val_acc_save_name)



        #Best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())


        # Sauvegarde
        if path_save is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                "history": history,
                "best_acc": best_acc,
                "best_model_weights": best_model_weights,
                "lr_scheduler_state": scheduler.state_dict()

            }, path_save)

            #print("Epoch {} sauvegardée".format(epoch))







if __name__ == "__main__":
    # generate_random_dataset_and_loader(250,100,100)
    pass