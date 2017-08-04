import numpy as np
import theano
import theano.tensor as TT

def get_ift_np(K, x, pi_min, pi_max):
    sinpiminx = np.sin(pi_min * x)
    sinpimaxx = np.sin(pi_max * x)
    cospiminx = np.cos(pi_min * x)
    cospimaxx = np.cos(pi_max * x)
    I_k_x = np.zeros((K+1, x.shape[0], 2))
    I_k_x[0,:,0] = (sinpimaxx-sinpiminx)/x
    I_k_x[0,:,1] = (cospimaxx-cospiminx)/x
    pi_min_k = pi_min
    pi_max_k = pi_max
    for k in range(1, K+1):
        I_k_x[k,:,0] = (sinpimaxx*pi_max_k-sinpiminx*pi_min_k)/x + (k/x) * I_k_x[k-1,:,1]
        I_k_x[k,:,1] = (cospimaxx*pi_max_k-cospiminx*pi_min_k)/x - (k/x) * I_k_x[k-1,:,0]
        pi_min_k = pi_min_k * pi_min
        pi_max_k = pi_max_k * pi_max
    return I_k_x

def get_ift_theano_build():
    K = TT.scalar("K", dtype="int32")
    x = TT.vector("x", dtype=theano.config.floatX)
    pi_min = TT.vector("pi_min", dtype=theano.config.floatX)
    pi_max = TT.vector("pi_max", dtype=theano.config.floatX)
    sinpimaxx = TT.sin(pi_max * x)
    sinpiminx = TT.sin(pi_min * x)
    cospimaxx = TT.cos(pi_max * x)
    cospiminx = TT.cos(pi_min * x)
    two = TT.constant(2., dtype=theano.config.floatX)
    one = TT.constant(1., dtype=theano.config.floatX)
    zero = TT.constant(0., dtype=theano.config.floatX)
    I_0_x_0 = (sinpimaxx-sinpiminx)/x
    I_0_x_1 = (cospimaxx-cospiminx)/x
    pi_min_1 = pi_min
    pi_max_1 = pi_max
    K_seq = TT.arange(K, dtype=theano.config.floatX)
    K_seq = K_seq + TT.ones_like(K_seq, dtype=theano.config.floatX)
    def get_ift_util(k, pi_min_k, pi_max_k, I_k_1_x_0, I_k_1_x_1, x, sinpimaxx, sinpiminx, cospimaxx, cospiminx, pi_max, pi_min):
        k_rep = TT.repeat(k, x.shape[0])
        I_k_x_0 = (sinpimaxx*pi_max_k-sinpiminx*pi_min_k)/x + (k_rep/x) * I_k_1_x_1
        I_k_x_1 = (cospimaxx*pi_max_k-cospiminx*pi_min_k)/x - (k_rep/x) * I_k_1_x_0
        pi_max_k = pi_max_k * pi_max
        pi_min_k = pi_min_k * pi_min
        return [pi_min_k, pi_max_k, I_k_x_0, I_k_x_1]
    scan_result, scan_updates = theano.scan(fn=get_ift_util,
                                        outputs_info=[pi_min_1, pi_max_1, I_0_x_0, I_0_x_1],
                                        sequences=K_seq,
                                        non_sequences=[x,sinpimaxx, sinpiminx,cospimaxx, cospiminx,pi_max, pi_min])
    scan_result_1 = TT.concatenate([I_0_x_0.dimshuffle('x',0), scan_result[1]], axis=0)
    scan_result_2 = TT.concatenate([I_0_x_1.dimshuffle('x',0), scan_result[2]], axis=0)
    result = TT.concatenate([scan_result_1.dimshuffle(0,1,'x'), scan_result_2.dimshuffle(0,1,'x')], axis=2)
    return theano.function([K,x,pi_min,pi_max], result)

get_ift_theano = get_ift_theano_build()
# get_ift_theano(5, np.asarray([1,1]).astype("float32"))