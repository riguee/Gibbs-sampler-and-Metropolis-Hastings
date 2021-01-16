import numpy as np
import pandas as pd
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# In[59]:

#open dataset and declare explainatory variables (indep)
dataset = pd.read_csv("Mroz.csv")
indep = ["k5", "k618", "age"]
cols = ['intercept']+indep
dataset = dataset.drop(columns=["Unnamed: 0"])
Y = dataset.hc.map({'yes': 1, 'no': 0})
X = dataset[indep].apply(pd.to_numeric)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
ones = np.ones((X.shape[0],1))
X = np.hstack((ones,X))

'''
# just for checks
from statsmodels.discrete.discrete_model import Probit

model = Probit(Y, X.astype(float))
probit_model = model.fit()
print(probit_model.summary())
'''


class Checking:
    """Class for keeping track of the program's trace"""
    def __init__ (self):
        self.Z = []
        self.betas= []

    def append(self, b, z):
        self.Z.append(z)
        self.betas.append(b)


# In[61]:


#chose prior to be normal with norm being a vector of ones and sigma the identity matrix
def update_beta(Z, X):
    #beta knowing Z using bayesian
    mu = np.linalg.inv(np.eye(X.shape[1]) + X.T @ X)@(np.ones(X.shape[1])+X.T@Z)
    sigma = np.linalg.inv(np.eye(X.shape[1]) + X.T@X)
    return np.random.multivariate_normal(mu, sigma)

def update_z(beta, X, Y):
    # for each Z, so X vector, Y scalar
    #Y==1 iff Z>0
    if Y==0:
        return stats.truncnorm.rvs(-np.inf, -X@beta, loc=X@beta)
    else:
        return stats.truncnorm.rvs(-X@beta, np.inf, loc=X@beta)

def gibbs_step(beta, X, Z, Y):
    new_beta = update_beta(Z, X)
    new_Z = np.zeros(len(Z))
    for i in range(len(Z)):
        new_Z[i] = update_z(beta, X[i], Y[i])
    return new_beta, new_Z

def gibbs(X, Y, iters):
    d = Checking()
    beta = np.random.multivariate_normal(np.ones(X.shape[1]), np.eye(X.shape[1]))
    Z = np.zeros(X.shape[0])
    for i in range(len(Z)):
        Z[i] = update_z(beta, X[i], Y[i])
    d.append(beta, Z)
    for i in range(iters):
        beta, Z = gibbs_step(beta, X, Z, Y)
        d.append(beta, Z)
        #keeping track of progress
        progress = np.linspace(0, int(iters), 101)
        if i in progress:
            print(f"\tProgress: {int(i / (iters / 100))} % ", end='\r')
    print('\tAll iterations of the Gibbs sampler are done')
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
        #print(key,betas_dist[key])
    return betas_dist, betas_dict

def diagnostics(indep, dep, iter, n_boostrap):

    tmp = gibbs(indep, dep, iter)
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

    plt.axhline(-0.2790, color="black") # intercept
    plt.axhline(0.1211, color="blue") #k5
    plt.axhline(-0.0318, color="green") #k618
    plt.axhline(-0.1484, color="red")  #age

    #plt.annotate("black = intercept\nblue = k5\ngreen = k618\nred = age",(0,0.2))
    plt.show()

    for k in b_dict.keys():
        plt.figure()
        plt.title(f'Bootstrap distribution of {k}')
        plt.hist(b_dict[k], density=True, label="samples", alpha=.5)
        plt.axvline(b_dist[k]['mean'], label="mean")
        plt.plot([b_dist[k]['lb_cred'], b_dist[k]['ub_cred']], [0,0], linewidth = 4, label="credibility interval", marker=".", markersize=15)
        plt.legend()
        plt.savefig(f"{k}_hist_gibbs.png")
        plt.show()

diagnostics(X, Y, 1000, 100)

'''
                          Probit Regression Results
==============================================================================
Dep. Variable:                     hc   No. Observations:                  753
Model:                         Probit   Df Residuals:                      749
Method:                           MLE   Df Model:                            3
Date:                Thu, 07 Jan 2021   Pseudo R-squ.:                 0.02249
Time:                        14:14:35   Log-Likelihood:                -492.82
converged:                       True   LL-Null:                       -504.16
Covariance Type:            nonrobust   LLR p-value:                 4.720e-05
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -0.2790      0.047     -5.968      0.000      -0.371      -0.187
x1             0.1211      0.052      2.328      0.020       0.019       0.223
x2            -0.0318      0.051     -0.629      0.529      -0.131       0.067
x3            -0.1484      0.056     -2.641      0.008      -0.259      -0.038
==============================================================================
'''
#tmp_betas.shape
