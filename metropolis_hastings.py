import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class Checking:
    """Class for keeping track of the program's trace"""
    def __init__ (self):
        self.a = []
        self.accept_state = []
        self.betas= []

    def append(self, a, s, b):
        self.a.append(a)
        self.accept_state.append(s)
        self.betas.append(b)


dataset = pd.read_csv("Mroz.csv")
dataset = dataset.drop(columns=["Unnamed: 0"])
Y = dataset.hc.map({'yes': 1, 'no': 0})
X = dataset[[ "k5", "k618","age"]].apply(pd.to_numeric)
scaler = StandardScaler()
scaler.fit(X)
ones = np.ones((X.shape[0],1))
X = scaler.transform(X)
X = np.hstack((ones,X))


def pdf_norm(v, mu, sigma):
    n = len(v)
    s = 1/np.linalg.det(sigma)
    e = np.exp(-((v-mu)@np.linalg.inv(sigma)@(v-mu))/2)
    return 1/np.sqrt(2*np.pi)**(n/2)*s*e

def pi(beta):
    return pdf_norm(beta, np.zeros(len(beta)), np.eye(len(beta)))

def phi(a):
    return stats.norm.cdf(a)

def phi_prime(a):
    return stats.norm.pdf(a)

def likelihood(beta, Y, X):
    l = 1
    for i, v in (enumerate(Y)):
        if v == 1:
            l *= phi(np.dot(X[i],beta))
        else:
            l *= (1 - phi(np.dot(X[i],beta)))

    return l

def inv_fisher(X, beta):
    w = [phi_prime(np.dot(X[i], beta))**2/(phi(np.dot(X[i], beta))*(1-phi(np.dot(X[i], beta)))) for i in range(X.shape[0])]
    W = np.diag(w)
    return (np.linalg.inv(X.T @ W @ X))

def alpha(beta, Y, X, beta_s, tau, pos):
    q1 = pdf_norm(beta_s, beta, tau * inv_fisher(X, beta))
    q2 = pdf_norm(beta, beta_s, tau * inv_fisher(X, beta_s))
    p = pi(beta_s) * likelihood(beta_s, Y, X)
    return (min(1, (p*q2)/(q1*pos)), p)

def accept(X, beta, Y, tau, pos):
    u = stats.uniform.rvs()
    beta_s = np.random.multivariate_normal(beta, tau * inv_fisher(X, beta))
    a, p = alpha(beta, Y, X, beta_s, tau, pos)
    if u < a:
        return a, p, beta_s, 1
    else:
        return a, pos, beta, 0

def metropolis(X, Y, tau, iters):
    i = 0
    d = Checking()
    # init posterior and beta
    beta = stats.norm.rvs(size=(X.shape[1]))
    pos = pi(beta) * likelihood(beta, Y, X)
    while i < iters:
        if i%10 == 0:
            print(f"doing iteration {i}", end="\r")
        a, pos, beta, s = accept(X, beta, Y, tau, pos)
        d.append(a, s, beta)
        i += 1
    return d



def bootstrap(betas, iterations, burn_in):
    betas = betas[burn_in:, :]
    betas_dict = {}
    betas_dist = {}
    estimators = ["intercept"] + indep

    for beta_i in estimators:
        betas_dict["{}".format(beta_i)] = []
        betas_dist["{}".format(beta_i)] = {"mean": "", "std": "", "lb_cred": "","ub_cred": ""}
        for i in range(iterations):
            betas_dict["{}".format(beta_i)].append(np.mean(random.choices(betas[:,estimators.index(beta_i)], k=100)))

        betas_dist["{}".format(beta_i)]["mean"] = np.mean(betas_dict["{}".format(beta_i)])
        betas_dist["{}".format(beta_i)]["std"] = np.std(betas_dict["{}".format(beta_i)])

        betas_dist["{}".format(beta_i)]["lb_cred"] = sorted(betas_dict["{}".format(beta_i)])[
            int(0.05 * iterations)]
        betas_dist["{}".format(beta_i)]["ub_cred"] = sorted(betas_dict["{}".format(beta_i)])[
            int(0.95 * iterations)]

    df = pd.DataFrame(columns=['variable', 'mean', 'std', 'lb_cred', 'ub_cred'])
    for key in estimators:
        betas_dist[key]['variable']=key
        df = df.append(betas_dist[key], ignore_index=True)
    df = df.set_index('variable')
    print("\n\n", df)
    return betas_dist, betas_dict

def diagnostics(indep, dep, iter, n_boostrap):

    tmp = metropolis(X,Y,1,100)
    tmp_betas = np.array(tmp.betas)
    burn_in = int(iter * 0.2)
    b_dist, b_dict = bootstrap(tmp_betas, n_boostrap, burn_in)

    '''
    for i in range(tmp_betas.shape[1]):
        plt.plot(np.arange(tmp_betas.shape[0]), tmp_betas[:, i], alpha=0.5)
    plt.show()
    indep = ["k5", "k618", "age"]
    '''
    colours = ["black", "blue", "green", "red", "orange", "yellow"]

    for i in range(X.shape[1]):
        plt.plot(np.arange(iter + 1 -burn_in), tmp_betas[burn_in:, i], alpha=0.5, label = f'{cols[i]}', color=colours[i]) #intercept
    plt.show()
    
    for k in b_dict.keys():
        plt.figure()
        plt.title(f'Bootstrap distribution of {k}')
        plt.hist(b_dict[k], density=True, label="samples", alpha=.5)
        plt.axvline(b_dist[k]['mean'], label="mean")
        plt.plot([b_dist[k]['lb_cred'], b_dist[k]['ub_cred']], [0,0], linewidth = 4, label="credibility interval", marker=".", markersize=15)
        plt.legend()
        plt.savefig(f"{k}_hist_mcmc.png")
        plt.show()



diagnostics(X, Y, 1000, 100)


