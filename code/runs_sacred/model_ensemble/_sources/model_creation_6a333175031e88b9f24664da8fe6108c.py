from torchvision.models import resnet18
import torch.nn as nn
from trainning import load_model_weights
import torch.nn.functional as F

import torch

def create_model(use_gpu,path_existing_model=None):

    model = resnet18(pretrained=True)
    model.fc = nn.Linear(512, 340)
    model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False) #1 canal


    #Pour geler
    # for param_name, param in model.named_parameters():
    #     if param_name not in ["fc.bias", "fc.weight"]:
    #         param.requires_grad = False


    # if path_existing_model is not None:
    #     model, history=load_model_weights(model,path_existing_model,use_gpu,type="last")


    return model



class Model_Ensemble(nn.Module):
    def __init__(self, list_model_saves_path,use_gpu,freeze_all):
        super(Model_Ensemble, self).__init__()

        model = create_model(use_gpu)
        self.model_0 = load_model_weights(model, list_model_saves_path[0], use_gpu=use_gpu)
        model = create_model(use_gpu)
        self.model_1 = load_model_weights(model, list_model_saves_path[1], use_gpu=use_gpu)

        if freeze_all:
            for param_name, param in self.model_0.named_parameters():
                param.requires_grad = False
            for param_name, param in self.model_1.named_parameters():
                param.requires_grad = False


        self.classifier = nn.Linear(self.nb_model*340, 340)

    def forward(self, x):
        list_output = []
        list_output[0]=self.model_0(x)
        list_output[1]=self.model_1(x)

        x = torch.cat(list_output, dim=1)
        x = self.classifier(x)


        return x




class Model_Ensemble_moyenne(nn.Module):
    def __init__(self, list_model_saves_path,use_gpu=True):
        super(Model_Ensemble_moyenne, self).__init__()

        model = create_model(use_gpu)
        self.model_0 = load_model_weights(model, list_model_saves_path[0], use_gpu=use_gpu)
        model = create_model(use_gpu)
        self.model_1 = load_model_weights(model, list_model_saves_path[1], use_gpu=use_gpu)



    def forward(self, x):

        y=(self.model_0(x) +self.model_1(x))/2




        return y



def create_ensemble_model(list_model_saves_path,use_gpu,frezze_all=True  ):


    model_ensemble=Model_Ensemble(list_model_saves_path,use_gpu,freeze_all=frezze_all)
    if use_gpu:
        model_ensemble.cuda()
    return model_ensemble



def create_ensemble_model_moy(list_model_saves_path,use_gpu):



    model_ensemble = Model_Ensemble_moyenne(list_model_saves_path,use_gpu)
    if use_gpu:
        model_ensemble.cuda()
    return model_ensemble




