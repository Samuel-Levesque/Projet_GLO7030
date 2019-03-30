import pickle

def save_object(object,save_name):
    with open(save_name, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(save_name):
    with open(save_name, 'rb') as handle:
        object = pickle.load(handle)

    return object


