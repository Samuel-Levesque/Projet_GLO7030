from utility import  load_object

dict_acc_general=load_object("dict_acc_per_class_valid_model_general.pk")
dict_acc_cible=load_object("dict_acc_per_class_valid_model_mauvaises_classes_sampling.pk")

def print_sorted_dict(dictionnaire,limit=0):
    for key, value in sorted(dictionnaire.items(), key=lambda item: item[1]):
        if value>limit:
            print("%s: %s" % (key, value))

# print_sorted_dict(dict_acc_general)
print_sorted_dict(dict_acc_cible)