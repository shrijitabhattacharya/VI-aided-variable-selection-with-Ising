import numpy as np
import torch
from sklearn import linear_model
from matplotlib import pyplot as plt
import time 

## Data loading
def read_data(fname):
    data=open(fname,'r')
    M=list()
    for line in data:
       m0=str.split(line,',')
       m0[-1]=str.split(m0[-1],'\n')[0]
       M.append([float(item) for item in m0])
    M=np.array(M)
    return M

def param_init(y, X, type,alpha=0.1):
    if type=='lasso':
        clf = linear_model.Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
        X0=np.array(X)
        y0=np.array(y)
        clf.fit(X0,y0)
        beta0=torch.tensor(clf.coef_)
    elif type=='ridge':
        clf = linear_model.Ridge(alpha=alpha, fit_intercept=False, max_iter=10000)
        X0=np.array(X)
        y0=np.array(y)
        clf.fit(X0,y0)
        beta0=torch.tensor(clf.coef_)
    return beta0

def kl_overall(gamma,mu,eta,yy,XXI,Xy,XXJI,tau,n,m,rho):
    s=torch.log(1+torch.exp(rho))
    phi=(1+gamma)/2
    sigma=torch.log(1+torch.exp(eta))
    val0=yy
    val1=torch.sum(torch.matmul(XXI,(mu**2+sigma**2)*phi))
    val2=torch.sum(Xy*phi*mu)
    val3=torch.sum(phi*mu*torch.matmul(XXJI,phi*mu))
    R1=yy+val1-2*val2+val3
    R2=torch.sum((mu**2+sigma**2)*phi)/tau**2
    R3=torch.sum((torch.log(tau**2/sigma**2)-1)*phi)
    R4=torch.sum(phi)
    ## E(-log L(beta,sigma))
    V1=0.5*torch.exp(-m+s**2/2)*R1+0.5*m*n
    ## E(q||p) beta
    V2=0.5*torch.exp(-m+s**2/2)*R2+0.5*R3+0.5*m*R4
    ## E(q||p) sigma
    V3=-torch.log(s)
    return V1+V2+V3

def grad_m_rho(gamma,mu,eta,yy,XXI,Xy,XXJI,n,m,rho):
    phi=(1+gamma)/2
    s=torch.log(1+torch.exp(rho))
    sigma=torch.log(1+torch.exp(eta))
    f=torch.nn.Sigmoid()
    val0=yy
    val1=torch.sum(torch.matmul(XXI,(mu**2+sigma**2)*phi))
    val2=torch.sum(Xy*phi*mu)
    val3=torch.sum(phi*mu*torch.matmul(XXJI,phi*mu))
    R1=val0+val1-2*val2+val3
    R2=torch.sum((mu**2+sigma**2)*phi)/tau**2
    R3=torch.sum((torch.log(tau**2/sigma**2)-1)*phi)
    R4=torch.sum(phi)
    dV1dm=-0.5*torch.exp(-m+s**2/2)*R1+0.5*n
    dV2dm=-0.5*torch.exp(-m+s**2/2)*R2+0.5*R4
    dm=dV1dm+dV2dm
    dV1drho=0.5*s*torch.exp(-m+s**2/2)*R1
    dV2drho=0.5*s*torch.exp(-m+s**2/2)*R2
    dV3drho=-1.0/s
    drho=(dV1drho+dV2drho+dV3drho)*f(rho)
    return dm, drho


def grad_mu(gamma,mu,XXI,Xy,XXJI,tau,m,rho):
    phi=(1+gamma)/2
    s=torch.log(1+torch.exp(rho))
    val1=torch.matmul(XXI,phi*mu)
    val2=Xy*phi
    val3=phi*torch.matmul(XXJI,phi*mu)
    val4=mu*phi/tau**2
    grad=val1-val2+val3+val4
    return grad*torch.exp(-m+s**2/2)

def grad_eta(gamma,eta,XXI,tau,m,rho):
    phi=(1+gamma)/2
    s=torch.log(1+torch.exp(rho))
    sigma=torch.log(1+torch.exp(eta))
    val1=torch.matmul(XXI,phi*sigma)
    val2=sigma*phi/tau**2
    val3=(1/sigma)*phi
    f=torch.nn.Sigmoid()
    grad=(val1+val2-val3)*f(eta)
    return grad*torch.exp(-m+s**2/2)

def gibbs_gamma(gamma,i,eta,mu,XXI,Xy,XXJI,B,a0,b0,n,m,rho):
    phi=(1+gamma)/2
    s=torch.log(1+torch.exp(rho))
    sigma=torch.log(1+torch.exp(eta))
    f=torch.nn.Sigmoid()
    val1=XXI[i,i]*(mu[i]**2+sigma[i]**2)
    val2=Xy[i]*mu[i]
    val3=torch.sum(XXJI[i,:]*(phi*mu))*mu[i]
    val4=(mu[i]**2+sigma[i]**2)/tau**2
    V1=(val1-2*val2+val3+val4)*torch.exp(-m+s**2/2)
    V2=torch.log(tau**2/sigma[i]**2)-1+m
    V3=b0*torch.sum(B[i,:]*gamma)+a0
    grad=-0.5*V1-0.5*V2+2*V3
    return f(grad)  

def adam_adjusted(t,mt,vt,gt,epsilon,decay1,decay2):
    mt=decay1*mt+(1-decay1)*gt
    vt=decay2*vt+(1-decay2)*gt**2
    mt_hat=mt/(1-decay1**t)
    vt_hat=vt/(1-decay2**t)
    return mt, vt, mt_hat/(torch.sqrt(vt_hat)+epsilon)

######################################################
## Covariates and response
X= read_data('X_NeurIPS_indep_p1000_n100.txt')
# X= read_data('X_NeurIPS_indep_rp20_cp50_n100.txt') # for image analysis
X= torch.tensor(X)
Ys= read_data('Y_NeurIPS_indep_beta21_p1000_n100.txt')
# Ys= read_data('Y_NeurIPS_indep_beta84_rp20_cp50_n100.txt') # for imange analysis

p= X.shape[1]
n= X.shape[0]

## True beta
beta= read_data('beta_21_NeurIPS_p1000.txt')
# beta= read_data('beta_84_NeurIPS_rp20_cp50.txt') # for image analysis
beta=torch.tensor(beta[:,0])

## Coupling matrix
B= read_data('Coupling_matrix_NeurIPS_p1000.txt')
# B= read_data('Coupling_matrix_NeurIPS_rp20_cp50.txt') # for image analysis
B= torch.tensor(B)

np.random.seed(1)
torch.manual_seed(1)

## Hyper parameters
d=2
r= 0.03
w1= 7 # w=1, 5, 7
w0= r*w1 + 1 - r

# 0/1 domain
a= np.log(r/(w0**2)) 
b= np.log(w1*w0) 

# -1 / 1 domain
a0= 0.5*a + 0.25*d*b # - 1.7533 / -1.0052 / - 0.863
b0= 0.25*b # 0.0 / 0.4307 / 0.528

tau= 0.3 

lr, epochs, burns, epsilon = 1e-3, 2000, 1, torch.tensor(1e-8) 
I=torch.eye(p)
J=torch.ones((p,p))
XX= torch.matmul(torch.transpose(X,0,1),X)

XXI=XX*I
XXJI=XX*(J-I)

I=torch.eye(p)
J=torch.ones((p,p))
supp_true=np.where(np.array(beta)!=0)[0]

## Initializations
eta_ini=torch.log(torch.exp(torch.tensor(0.127))-1)*torch.ones((p,))
eta_ini=eta_ini.type(torch.DoubleTensor)
m_ini= 0.0 
rho_ini=torch.log(torch.exp(torch.tensor(0.127))-1)
gamma_ini=2*torch.bernoulli(0.5*torch.ones((p,)))-1 
gamma_ini=gamma_ini.type(torch.DoubleTensor)
prob=torch.zeros((p,))
prob=prob.type(torch.DoubleTensor)
eta, m, rho, gamma = eta_ini, m_ini, rho_ini, gamma_ini

### Running the algorithm
## 10 replicates
R=10
post_prob_sum_iter10= torch.zeros(size=(R,p))
times_iter10= torch.zeros(R)
for rep in range(R):
    before = time.time()
    
    y=torch.tensor(Ys[:,rep])
    Xy= torch.matmul(torch.transpose(X,0,1),y)
    yy=torch.sum(y*y)
    mu_ini=param_init(y,X, 'lasso', alpha=0.01) # alpha
    mu_ini=mu_ini.type(torch.DoubleTensor)
    mu = mu_ini
    
    mt_mu, vt_mu=torch.zeros((p,)),torch.zeros((p,))
    mt_eta, vt_eta=torch.zeros((p,)),torch.zeros((p,))
    mt_m, vt_m=torch.tensor(0.0), torch.tensor(0.0)
    mt_rho, vt_rho=torch.tensor(0.0), torch.tensor(0.0)
    decay1, decay2=torch.tensor(0.9), torch.tensor(0.999)
    gammaS=torch.zeros((epochs,p))
    for t in range(epochs):
        kl=kl_overall(gamma,mu,eta,yy,XXI,Xy,XXJI,tau,n,m,rho)
        supp_pred=np.where(np.array((gamma+1)/2)!=0)[0]
        fs = set(supp_pred).difference(set(supp_true)) # false selection number
        ns = set(supp_true).difference(set(supp_pred)) # negative selection number
        # print(kl.item(),len(fs),len(ns))
        gt_mu=grad_mu(gamma,mu,XXI,Xy,XXJI,tau,m,rho)
        mt_mu, vt_mu, gt_mu=adam_adjusted((t+1),mt_mu,vt_mu,gt_mu,epsilon,decay1,decay2)   
        mu=mu-lr*gt_mu
        gt_eta=grad_eta(gamma,eta,XXI,tau,m,rho)
        mt_eta, vt_eta, gt_eta=adam_adjusted((t+1),mt_eta,vt_eta,gt_eta,epsilon,decay1,decay2)
        eta=eta-lr*gt_eta
        sigma=torch.log(1+torch.exp(eta))
        gt_m, gt_rho=grad_m_rho(gamma,mu,eta,yy,XXI,Xy,XXJI,n,m,rho)
        mt_m, vt_m, gt_m=adam_adjusted((t+1),mt_m,vt_m,gt_m,epsilon,decay1,decay2)   
        m=m-lr*gt_m
        mt_rho, vt_rho, gt_rho=adam_adjusted((t+1),mt_rho,vt_rho,gt_rho,epsilon,decay1,decay2)   
        rho=rho-lr*gt_rho
        s=torch.log(1+torch.exp(rho))
        for u in range(burns): # 1
            for i in range(p):
                prob[i]=gibbs_gamma(gamma,i,eta,mu,XXI,Xy,XXJI,B,a0,b0,n,m,rho)
                gamma[i]=2*torch.bernoulli(prob[i])-1
        gammaS[t,:]=gamma
    
    gammaS= gammaS[int(epochs/2):,] 

    phiS=(gammaS+1)/2 
    post_prob_sum=torch.sum(phiS,0) 
    
    after = time.time()
    print("time")
    print(after - before) 
    
    post_prob_sum_iter10[rep,:]= post_prob_sum
    times_iter10[rep]= after - before

# torch.save(post_prob_sum_iter10, 'post_prob_sum_VB_beta21_indepX_p1000_n100_w7_tau03_lr1e-3_alp001_iter10.pt') 

