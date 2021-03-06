from torchvision.models import resnet18
import torch.nn as nn



def create_model(use_gpu,path_existing_model=None):

    model = resnet18(pretrained=True)
    model.fc = nn.Linear(512, 340)
    model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False) #1 canal

    for param_name, param in model.named_parameters():
        if param_name not in ["fc.bias", "fc.weight"]:
            param.requires_grad = False


    # if path_existing_model is not None:
    #     model, history=load_best_model(model,path_existing_model,use_gpu,use_model_last_epoch=True)


    return model