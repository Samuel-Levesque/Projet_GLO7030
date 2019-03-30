from torch.utils.data import DataLoader
from model_creation import create_model
from trainning import load_model_weights
from torch.autograd import Variable
from data_set_file import DoodlesDataset
from utility import load_object
import pandas as pd



def prediction_data(loader,path_save_model,use_gpu):


    # Model
    model = create_model(use_gpu)
    if use_gpu:
        model.cuda()

    model,history=load_model_weights(model,path_save_model,type="last",use_gpu=use_gpu,get_history=True)


    model.train(False)


    pred_top3=[]
    model.eval()

    for j, inputs in enumerate(loader):

        if type(inputs)is list:
            inputs=inputs[0]

        if use_gpu:
            inputs = inputs.cuda()


        inputs = Variable(inputs, volatile=True)

        output = model(inputs)


        predictions_top_3 = output.topk(3)[1]

        pred_top3.extend(predictions_top_3.data.cpu().numpy().tolist())



    model.train(True)
    return pred_top3


#Pour faire prédiction pour soumettre sur kaggle
if __name__ == "__main__":
    path_save_model = "saves_model/model_info.tar"
    use_gpu=True
    path = 'D:/User/William/Documents/Devoir/Projet Deep/data/test/'
    csv_file = "test_simplified.csv"
    decoding_dict=load_object("saves_obj/dec_dict.pk")


    data_set = DoodlesDataset(csv_file, path, mode="test",nrows=112200)
    loader = DataLoader(data_set, batch_size=1)


    list_top3=prediction_data(loader,path_save_model,use_gpu=use_gpu)

    list_top3_string=[decoding_dict[top3[0]]+" " + decoding_dict[top3[1]]+" " + decoding_dict[top3[2]]  for top3 in list_top3]
    data_pred= pd.read_csv(path+csv_file, usecols=['key_id'])

    data_pred["word"]=list_top3_string

    data_pred.to_csv(path+"test_simplified_pred.csv", sep=',', encoding='utf-8')


    pass

