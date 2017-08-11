import numpy as np
np.random.seed(2)
import matplotlib.pyplot as plt

def phi_t(t, moments):
    val = 0
    num_coef = 1.
    denom_coef = 1.
    cnt = 1
    for m in moments:
        val += (m * num_coef) / denom_coef
        num_coef *= (t*1j)
        denom_coef *= cnt
        cnt += 1
    return val

def get_moments(X, K):
    val = []
    m = np.ones(X.shape)
    for i in range(K):
        val.append(np.mean(m))
        m *= X
    return val

K = 20
X = np.random.normal(0., 1., (100000))
moments = get_moments(X, K)
t_vals = np.linspace(-3., 3., 1000)
N = t_vals.shape[0]
phi_t_vals = []
for i in range(N):
    phi_t_vals.append(phi_t(t_vals[i], moments))

plt.plot(t_vals, np.real(np.asarray(phi_t_vals)), "ro")
plt.plot(t_vals, np.exp(-0.5*(np.asarray(t_vals)**2)), "bo")
plt.show()