from sacred import  Experiment
from sacred.observers import FileStorageObserver



#Trucs sacred
experiment_sacred=Experiment("Doodle_Boys")
experiment_sacred.observers.append(FileStorageObserver.create('my_runs_v_alpha'))



#Configs
@experiment_sacred.config
def configuration():
    path_data = 'D:/User/William/Documents/Devoir/Projet Deep/data/mini_train/'

    pass




#Main
@experiment_sacred.automain
def main_program(path_data):
    print(path_data)
    pass



