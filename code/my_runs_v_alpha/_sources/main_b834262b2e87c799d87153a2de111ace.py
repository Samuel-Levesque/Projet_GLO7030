from sacred import  Experiment
from sacred.observers import FileStorageObserver
from data_set_file import create_huge_data_set,create_encoding_deconding_dict
from model_creation import create_model
from trainning import  LRPolicy,train_model,load_model_weights,create_scheduler

from deeplib.training import validate




import torch.optim as optim
import torch.nn as nn

from torch.optim.lr_scheduler import  LambdaLR
from torch.utils.data import DataLoader

#Trucs sacred
experiment_sacred=Experiment("Doodle_Boys")
experiment_sacred.observers.append(FileStorageObserver.create('my_runs_v_alpha'))



#Configs
@experiment_sacred.config
def configuration():

    path_data = 'D:/User/William/Documents/Devoir/Projet Deep/data/mini_train/'
    path_save_model="saves_model/model_info.tar"
    use_gpu = True


    #ATTENTION CHANGER NB CATEGORIE MODEL

    do_training=True
    do_testing=True



    nb_row_per_classe=300




    nb_epoch = 3
    batch_size = 32

    learning_rate = 0.1
    type_schedule="constant"





    




#Main
@experiment_sacred.automain
def main_program(path_data,nb_row_per_classe,
                 use_gpu,do_training,do_testing,
                 nb_epoch,batch_size,
                 learning_rate,type_schedule,
                 path_save_model):




    # Label encoding and decoding dicts
    enc_dict, dec_dict = create_encoding_deconding_dict(path_data)




    #Data_set
    size_image_train = 224
    data_train=create_huge_data_set(path_data,nb_rows=nb_row_per_classe,size_image=size_image_train,encoding_dict=enc_dict)

    data_valid=create_huge_data_set(path_data,nb_rows=100,size_image=size_image_train,skip_rows=range(1,nb_row_per_classe),encoding_dict=enc_dict)
    data_test = create_huge_data_set(path_data, nb_rows=100, size_image=size_image_train,skip_rows=range(1, nb_row_per_classe+100), encoding_dict=enc_dict)





    # Model
    model = create_model(use_gpu)

    if use_gpu:
        model.cuda()

    #Loss
    criterion = nn.CrossEntropyLoss()

    #Optimiser
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Scheduler LR
    scheduler = create_scheduler(start_lr=learning_rate,type=type_schedule)


    #Data loader
    train_loader=DataLoader(data_train,batch_size=batch_size,shuffle=True)
    valid_loader=DataLoader(data_valid,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(data_test,batch_size=batch_size)



    #Train
    if do_training:
        train_model(model,train_loader,valid_loader,nb_epoch,
                    scheduler,optimizer,criterion,use_gpu,
                    path_save=path_save_model)





    #Test
    if do_testing:
        print("BIDON")
        model_final,history=load_model_weights(model,path_save_model,type="last",use_gpu=use_gpu,get_history=True)
        history.display()
        score = validate(model_final, test_loader, use_gpu=use_gpu)[0]
        print("Accuracy test:",score)





    pass



