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
    def __init__(self, list_models):
        super(Model_Ensemble, self).__init__()


        self.nb_model=len(list_models)
        self.list_models =list_models



        self.classifier = nn.Linear(self.nb_model*340, 340)

    def forward(self, x):
        list_output = []
        for model in self.list_models:
            output = model(x)
            list_output.append(output)

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

        y=self.model_0(x)




        return y



def create_ensemble_model(list_model_saves_path,use_gpu,frezze_all=True  ):
    list_model=[]
    for model_save_path in list_model_saves_path:
        model=create_model(use_gpu)
        model=load_model_weights(model,model_save_path,use_gpu=use_gpu)

        if frezze_all:
            for param_name, param in model.named_parameters():
                param.requires_grad = False

        if use_gpu:
            model.cuda()
        list_model.append(model)

    model_ensemble=Model_Ensemble(list_model)
    return model_ensemble



def create_ensemble_model_moy(list_model_saves_path,use_gpu,frezze_all=True  ):



    model_ensemble = Model_Ensemble_moyenne(list_model_saves_path,use_gpu)
    if use_gpu:
        model_ensemble.cuda()
    return model_ensemble




