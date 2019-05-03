from utility import  load_object
import numpy as np

dict_acc_general=load_object("dict_acc_per_class_valid_model_general.pk")
dict_acc_cible=load_object("dict_acc_per_class_valid_model_mauvaises_classes_sampling.pk")

def print_sorted_dict(dictionnaire):
    low_bad_list=[]
    i=0
    for key, value in sorted(dictionnaire.items(), key=lambda item: item[1]):

        print("%s: %s" % (key, value))
        i+=1
        if i<=50:
            low_bad_list.append(value)

    print("Acc moyenne 20 pires: ",np.mean(low_bad_list))



# print_sorted_dict(dict_acc_general)
print_sorted_dict(dict_acc_cible)