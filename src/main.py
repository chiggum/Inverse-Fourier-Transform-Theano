import numpy as np
np.random.seed(2)

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
from dense_transpose import DenseTranspose
from orthogonal_regularizer import OrthogonalL1L2
from keras.callbacks import ModelCheckpoint
from keras import backend as KK
from compute_ift import get_ift_np, get_ift_theano

batch_size = 128
num_classes = 10
epochs = 100
ortho_l1 = 1.0
ortho_l2 = 0.0

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print x_train.shape[0], 'train samples'
print x_test.shape[0], 'test samples'

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)


inputs_ = Input(shape=(784,))
od_1 = Dense(256, kernel_regularizer=OrthogonalL1L2(ortho_l1, ortho_l2))
od_2 = Dense(128, kernel_regularizer=OrthogonalL1L2(ortho_l1, ortho_l2))
od_3 = Dense(64, kernel_regularizer=OrthogonalL1L2(ortho_l1, ortho_l2))
od_4 = Dense(32, kernel_regularizer=OrthogonalL1L2(ortho_l1, ortho_l2))
od_5 = Dense(16, kernel_regularizer=OrthogonalL1L2(ortho_l1, ortho_l2))
od_6 = Dense(8, kernel_regularizer=OrthogonalL1L2(ortho_l1, ortho_l2))

out_1 = od_1(inputs_)
out_2 = od_2(out_1)
out_3 = od_3(out_2)
out_4 = od_4(out_3)
out_5 = od_5(out_4)
out_6 = od_6(out_5)

out_7 = DenseTranspose(od_6, 16)(out_6)
out_8 = DenseTranspose(od_5, 32)(out_7)
out_9 = DenseTranspose(od_4, 64)(out_8)
out_10 = DenseTranspose(od_3, 128)(out_9)
out_11 = DenseTranspose(od_2, 256)(out_10)
out_12 = DenseTranspose(od_1, 784)(out_11)

model = Model(inputs=inputs_, outputs=out_12)

model.compile(loss='mean_squared_error',
              optimizer="adam",
              metrics=['mean_squared_error'])

out_6_fn = KK.function([inputs_], out_6)

# model.fit(x_train, x_train,
#             batch_size=batch_size,
#             epochs=epochs,
#             verbose=1,
#             validation_data=(x_test, x_test),
#             callbacks=[ModelCheckpoint("wts_2.h5", monitor='mean_squared_error', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)])

model.load_weights("wts.h5")
score = model.evaluate(x_test, x_test, verbose=1)
print('Test loss:', score[0])
print('Test mean_squared_error:', score[1])

z_train = out_6_fn([x_train])
z_test = out_6_fn([x_test])

print "z_train.shape", z_train.shape
print "z_test.shape", z_test.shape

#v is base
#p is number of digits
def sequence(v, p, column):
    subsequence = []
    for i in range(v):
        subsequence += [i] * v**(p - column)
    return subsequence * v**(column - 1)

def getEz(z, N):
    n_ex = z.shape[0]
    z_dim = z.shape[1]
    base = N+1
    digits = z_dim
    all_seq = []
    for i in range(digits):
        my_seq = sequence(base, digits, i+1)
        all_seq.append(my_seq)
    all_seq = np.asarray(all_seq)
    all_seq = np.transpose(all_seq)
    Ez = {}
    for i in range(N+1):
        Ez[i] = {}
    for i in range(all_seq.shape[0]):
        S = np.sum(all_seq[i,:])
        if S <= N:
            this_Ez = np.ones(n_ex)
            for j in range(all_seq.shape[1]):
                this_Ez = this_Ez * np.power(z[:,j],all_seq[i,j])
            Ez[S][tuple(all_seq[i,:])] = np.mean(this_Ez)
    return Ez

def xCy(N):
    output = np.zeros((N+1,N+1))
    for i in range(N+1):
        output[i,0] = 1
        output[i,1] = i
    for i in range(2, N+1):
        for j in range(1, N+1):
            output[i,j] = output[i-1,j] + output[i-1,j-1]
    return output

def get_coeffs(Ez, x_C_y):
    coeff = {}
    for k in Ez.keys():
        coeff[k] = {}
    for k,v in Ez.iteritems():
        for k2 in v.keys():
            k2_list = list(k2)
            multip = 1.
            n = k
            for i in range(len(k2_list)):
                multip *= x_C_y[n,k2_list[i]]
                n = n - k2_list[i]
            coeff[k][k2] = v[k2] * multip
    return coeff

def get_final_prob(Ez_coeff, ift_coeff):
    final_coeff = np.zeros((ift_coeff.shape[0],2))
    for k,v in Ez_coeff.iteritems():
        S1 = np.asarray([0.,0.])
        for k2,v2 in v.iteritems():
            S2 = np.asarray([1.,1.])
            for i_k3 in range(len(k2)):
                S2 = S2 * ift_coeff[k2[i_k3], i_k3, :]
            S1 += v2 * S2
        final_coeff[k,:] = S1
    return final_coeff

def get_factorial_vec(N):
    fact_vec = np.zeros(N+1)
    fact_vec[0] = 1
    for i in range(1,N+1):
        fact_vec[i] = fact_vec[i-1]*i
    return fact_vec

def compute_P_z(num_classes, y_train, z_train, y_test, z_test, N_moments=5):
    range_z = {}
    Ez_train = {}
    for c in range(num_classes):
        Ez_train[c] = getEz(z_train[y_train==c,:], N_moments)
        range_z[c] = (np.min(z_train[y_train==c,:], axis=0)*0-2.5, np.max(z_train[y_train==c,:], axis=0)*0+2.5)
    x_C_y = xCy(N_moments)
    fact_vec = get_factorial_vec(N_moments)
    coeffs = {}
    for c in range(num_classes):
        coeffs = get_coeffs(Ez_train[c], x_C_y)
    # not exact prob but proportional
    # exact will be their multiplication with (1/2pi)^{z_train.shape[1]}
    normalizer = np.power(2*np.pi,z_train.shape[1])
    P_z_train_given_y = np.zeros((z_train.shape[0], num_classes,2))
    P_z_test_given_y = np.zeros((z_test.shape[0], num_classes,2))
    idx_train = np.arange(z_train.shape[0])
    idx_test = np.arange(z_test.shape[0])
    np.random.shuffle(idx_train)
    np.random.shuffle(idx_test)
    for i_ in range(int(idx_train.shape[0]/100)):
        i = idx_train[i_]
        print "train:", i_, i
        ift_coeff = get_ift_theano(N_moments, z_train[i,:].astype("float32"), range_z[c][0], range_z[c][1])
        for c in range(num_classes):
            myP = get_final_prob(Ez_train[c], ift_coeff)
            P_z_train_given_y[i_, c, 0] = np.sum(myP[:,0]/fact_vec)/normalizer
            P_z_train_given_y[i_, c, 1] = np.sum(myP[:,1]/fact_vec)/normalizer
    for i_ in range(int(idx_test.shape[0]/10)):
        i = idx_test[i_]
        print "test:", i_, i
        ift_coeff = get_ift_theano(N_moments, z_test[i,:].astype("float32"), range_z[c][0], range_z[c][1])
        for c in range(num_classes):
            myP = get_final_prob(Ez_train[c], ift_coeff)
            P_z_test_given_y[i_, c, 0] = np.sum(myP[:,0]/fact_vec)/normalizer
            P_z_test_given_y[i_, c, 1] = np.sum(myP[:,1]/fact_vec)/normalizer
    y_train_new = y_train[idx_train[:int(idx_train.shape[0]/100)]]
    y_test_new = y_test[idx_test[:int(idx_test.shape[0]/10)]]
    return P_z_train_given_y, P_z_test_given_y, y_train_new, y_test_new

P_z_train_given_y, P_z_test_given_y, y_train_new, y_test_new = compute_P_z(num_classes, y_train, z_train, y_test, z_test, 8)