''''
pour partir de l'image du UI en png
'''

from torchvision.datasets import DatasetFolder,ImageFolder
import torchvision.transforms as transforms
from inference import prediction_data
from torch.utils.data import  DataLoader
from utility import load_object



def predict_image_classes(path_image_folder,path_save_model,use_gpu,decoding_dict):
    transformation = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


    data_set = ImageFolder(path_image_folder, transform=transformation)

    loader = DataLoader(data_set, batch_size=1)

    pred=prediction_data(loader,path_save_model,use_gpu)[0]

    pred_string=[decoding_dict[element] for element in pred]

    return pred_string




if __name__ == "__main__":
    path_save_model = "saves_model/model_info.tar"
    use_gpu=True



    path='D:/User/William/Documents/Devoir/Projet Deep/data/photo_inference/'
    decoding_dict=load_object("saves_obj/dec_dict.pk")
    predtiction_classe=predict_image_classes(path,path_save_model,use_gpu,decoding_dict)

    print(predtiction_classe)
