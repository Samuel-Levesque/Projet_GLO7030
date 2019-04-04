from sacred import  Experiment
from sacred.observers import FileStorageObserver
from data_set_file import create_huge_data_set,create_encoding_deconding_dict,generate_random_dataset
from model_creation import create_model
from trainning import  train_model,load_model_weights,create_scheduler
from test_metrics import calcul_metric_concours

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader

#Trucs sacred
experiment_sacred=Experiment("Doodle_Boys")
experiment_sacred.observers.append(FileStorageObserver.create('runs_sacred/model_data_random'))



#Configs
@experiment_sacred.config
def configuration():

    path_data = 'D:/User/William/Documents/Devoir/Projet Deep/data/mini_train/'

    path_save_model="saves_model/model_poids_random.tar"
    path_load_existing_model = "saves_model/model_poids_random.tar"
    path_model_weights_test = "saves_model/model_poids_random.tar"

    use_gpu = True


    do_training=True
    do_testing=True



    nb_row_per_classe = 100
    nb_row_class_valid=100
    nb_row_class_test=100
    skip_test=range(nb_row_class_valid)
    nb_generation_random_dataset_train=4




    nb_epoch = 3
    batch_size = 32

    learning_rate = 0.1
    type_schedule="constant"

    seed=123 #marche paas








#Main
@experiment_sacred.main
def main_program(path_data,path_save_model,path_load_existing_model,path_model_weights_test,
                 use_gpu,do_training,do_testing,
                 nb_row_per_classe,nb_generation_random_dataset_train,
                 nb_row_class_valid,nb_row_class_test,skip_test,

                 nb_epoch,batch_size,
                 learning_rate,type_schedule,
                 seed
                 ):


    #Seed
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    # Label encoding and decoding dicts
    enc_dict, dec_dict = create_encoding_deconding_dict(path_data)



    # Model
    model = create_model(use_gpu)

    if use_gpu:
        model.cuda()

    #Loss
    criterion = nn.CrossEntropyLoss()

    #Optimiser
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Scheduler LR
    scheduler = create_scheduler(start_lr=learning_rate,type=type_schedule,optimizer=optimizer)

    # Data_set
    size_image_train = 224
    # data_train = create_huge_data_set(path_data, nb_rows=nb_row_per_classe, size_image=size_image_train,encoding_dict=enc_dict)
    data_valid = create_huge_data_set(path_data, nb_rows=nb_row_class_valid, size_image=size_image_train, encoding_dict=enc_dict)

    #Data loader
    # train_loader=DataLoader(data_train,batch_size=batch_size,shuffle=True)
    valid_loader=DataLoader(data_valid,batch_size=batch_size,shuffle=True)




    #Train
    if do_training:

        for i in range(nb_generation_random_dataset_train):
            print("Data set {} utilisé pour entraînement".format(i))
            data_train=generate_random_dataset(path_data,nb_row_class_valid,nb_row_class_test,nb_row_per_classe,
                                               size_image=size_image_train,encoding_dict=enc_dict)

            train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)


            train_model(model,train_loader,valid_loader,nb_epoch,
                        scheduler,optimizer,criterion,use_gpu,
                        path_save=path_save_model,path_start_from_existing_model=path_load_existing_model)





    #Test
    if do_testing:
        data_test = create_huge_data_set(path_data, nb_rows=nb_row_class_test, size_image=size_image_train,
                                         skip_rows=skip_test, encoding_dict=enc_dict)
        test_loader = DataLoader(data_test, batch_size=batch_size)




        model_final,history=load_model_weights(model,path_model_weights_test,type="best",use_gpu=use_gpu,get_history=True)
        # history.display()

        acc,loss,score_top3,conf_mat,acc_per_class=calcul_metric_concours(model_final,test_loader,use_gpu=use_gpu,show_acc_per_class=True)

        print("Accuracy test: {}".format(acc))
        print("Score top 3 concours: {}".format(score_top3))
        print(acc_per_class)

        #Log experiment
        experiment_sacred.log_scalar("Test accuracy",acc)
        experiment_sacred.log_scalar("Test loss", loss)
        experiment_sacred.log_scalar("Test score top3", score_top3)
        experiment_sacred.log_scalar("Test confusion matrix", conf_mat)
        experiment_sacred.log_scalar("Test accuracy per class", acc_per_class)



experiment_sacred.run()








