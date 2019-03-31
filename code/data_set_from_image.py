''''
pour partir de l'image du UI en png
'''

from torchvision.datasets import DatasetFolder,ImageFolder
import torchvision.transforms as transforms
from inference import prediction_data
from torch.utils.data import  DataLoader
from utility import load_object
import time


def predict_image_classes(path_image_folder,path_save_model,use_gpu,decoding_dict,file_number=1):
    '''
    Permet de renvoyer une list top 3 prédit à partir d'une image png.

    ex:
    On a : la photo 00001.png;
    D:/User/William/Documents/Devoir/Projet Deep/data/photo_inference/dessin/00001.png





    :param path_image_folder:    D:/User/William/Documents/Devoir/Projet Deep/data/photo_inference/
    :param path_save_model:  Path des poids du modèle
    :param use_gpu:
    :param decoding_dict: dict pour décoder les classes de chiffres à string : {0: "airplane", 1: "alarm_clock",...}
    :return: list de 3 items, le premier est le plus probable, list de prob associé
    '''
    transformation = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


    data_set = ImageFolder(path_image_folder, transform=transformation)

    loader = DataLoader(data_set, batch_size=1)

    # pred=prediction_data(loader,path_save_model,use_gpu)[0]
    list_pred,list_tensor_prob=prediction_data(loader,path_save_model,use_gpu,get_prob_pred=True)


    pred=list_pred[file_number-1]
    prob=list_tensor_prob[file_number-1].data.numpy()*100



    pred_string=[decoding_dict[element] for element in pred]

    return pred_string,prob




if __name__ == "__main__":

    start_time=time.time()

    path_save_model = "saves_model/model_info.tar"
    use_gpu=False
    file_number=1



    path='D:/User/William/Documents/Devoir/Projet Deep/data/photo_inference/'
    decoding_dict=load_object("saves_obj/dec_dict.pk")
    predtiction_classe,prob=predict_image_classes(path,path_save_model,use_gpu,decoding_dict,file_number)

    print(predtiction_classe)
    print(prob)
    print("Temps de calcul: {} secondes".format(time.time()-start_time)) #Plus rapide sans GPU
