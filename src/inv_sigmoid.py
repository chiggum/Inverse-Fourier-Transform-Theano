from keras import backend as K

def inv_sigmoid(x):
    return K.log(x/(1-x))