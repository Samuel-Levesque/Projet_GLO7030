from sacred import  Experiment
from sacred.observers import FileStorageObserver
from data_set_file import create_huge_data_set


#Trucs sacred
experiment_sacred=Experiment("Doodle_Boys")
experiment_sacred.observers.append(FileStorageObserver.create('my_runs_v_alpha'))



#Configs
@experiment_sacred.config
def configuration():
    path_data = 'D:/User/William/Documents/Devoir/Projet Deep/data/mini_train/'

    nb_row_per_classe=400


    pass




#Main
@experiment_sacred.automain
def main_program(path_data,nb_row_per_classe):


    size_image_train = 224
    data_train=create_huge_data_set(path_data,nb_rows=nb_row_per_classe,size_image=size_image_train)

    data_valid=create_huge_data_set(path_data,nb_rows=nb_row_per_classe,size_image=size_image_train,skip_rows=nb_row_per_classe-1)


    print(path_data)
    pass



