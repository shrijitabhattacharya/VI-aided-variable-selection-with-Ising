import torch
import torch.distributions
import numpy as np
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

## Gibbs Sampling from posterior distribution
def posterior_sampling(a, b, burnin=50):
    initial_prob= 0.5*torch.ones(p) # 0.5 0.1
    gam= torch.bernoulli(initial_prob)

    gammas= torch.zeros(2*burnin, p)
    for step in range(2*burnin):
        
        for i in range(p):
            sum_of_neigs= torch.matmul(B[i,:].float(), gam.float())
        
            numer= torch.exp(a + b*sum_of_neigs)
            denom= 1 + torch.exp(a + b*sum_of_neigs)
            prior_prob= numer / denom
            
            temp= gam.clone()
            temp[i] = 0
            
            I_minus_i= torch.where(temp == 1)[0]                        
            p_minus_i= len(I_minus_i)
            X_I_minus_i= X[:,I_minus_i]
            X_I_minus_i_t= torch.transpose(X_I_minus_i,0,1)
            A_minus_i= torch.matmul(X_I_minus_i_t, X_I_minus_i) + (nu**(-2))*torch.eye(p_minus_i)
            A_minus_i_inv= torch.inverse(A_minus_i)
            
            # Fast matrix inversion
            Xi = X[:,i]
            Sig_Ii= torch.matmul(X_I_minus_i_t, Xi)
            sig_ii= torch.matmul(Xi, Xi) + nu**(-2)
            
            v1 = torch.matmul(torch.matmul(Sig_Ii, A_minus_i_inv), Sig_Ii)
            v2 = sig_ii - v1
            
            V1 = torch.matmul(A_minus_i_inv, Sig_Ii)
            V1 = V1.reshape(p_minus_i,1)
            V2 = torch.matmul(V1, torch.transpose(V1, 0, 1))
            
            A11 = A_minus_i_inv + (V2/v2)
            A12 = - torch.matmul(A11, Sig_Ii) / sig_ii
            
            a1= Sig_Ii / sig_ii
            A22= sig_ii**(-1) + torch.matmul(a1, torch.matmul(A11, a1))
            
            A_i_inv= torch.zeros(p_minus_i+1, p_minus_i+1)
            A_i_inv[:p_minus_i,:p_minus_i] = A11
            
            a2 = torch.zeros(p_minus_i + 1)
            a2[:p_minus_i] = A12
            a2[p_minus_i] = A22
            A_i_inv[p_minus_i,:] = a2
            A_i_inv[:p_minus_i,p_minus_i] = A12

            I_i= torch.zeros(len(I_minus_i)+1)
            I_i[:len(I_minus_i)]= I_minus_i
            I_i[len(I_minus_i)]= i
            p_i= len(I_i)
        
            X_I_i= X[:,I_i.long()]
            X_I_i_t= torch.transpose(X_I_i,0,1)
            A_i= torch.matmul(X_I_i_t, X_I_i) + (nu**(-2))*torch.eye(p_i)
            
            temp = -torch.log(torch.tensor(nu)) + 0.5*torch.logdet(A_minus_i) - 0.5*torch.logdet(A_i)
            t1 = torch.exp(temp)
            
            Y_t = torch.transpose(Y, 0, 1)
            term1 = torch.matmul(Y_t, Y)
            term2 = torch.matmul(torch.matmul(X_I_minus_i.float(), A_minus_i_inv.float()), X_I_minus_i_t.float())

            numer = term1 - torch.matmul(torch.matmul(Y_t.float(), term2.float()), Y.float())
            
            term3 = torch.matmul(torch.matmul(X_I_i.float(), A_i_inv.float()), X_I_i_t.float())
          
            denom = term1 - torch.matmul(torch.matmul(Y_t.float(), term3.float()), Y.float())
            t2 = (numer / denom)**(0.5*n)
            
            BF= t1*t2

            posterior_prob= (prior_prob) / ( prior_prob + ((1-prior_prob)/BF) )
            
            gam[i] = torch.bernoulli(posterior_prob)
        
        if (step+1)%500 == 0:
            print(step)
        gammas[step,:] = gam
        
    return gammas

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

## Hyper parameters
nu= 0.3 
burn_in= 1000

r= 0.03
w1= 7 # w=1, 5, 7
w0= r*w1 + 1 - r
a= np.log(r/(w0**2)) 
b= np.log(w1*w0) 

np.random.seed(1)
torch.manual_seed(1)

## 10 replicates
R= 10
post_prob_sum_iter10= torch.zeros(size=(R,p))
for rep in range(R):
    Y = torch.tensor(Ys[:,rep]).reshape(n,1)
    
    before = time.time()
    posterior_gammas = posterior_sampling(a, b, burnin=burn_in)  
    posterior_gammas = posterior_gammas[burn_in:,]
    after = time.time()
    print(after - before) 
    print(rep)
    marg_prob_sum= torch.sum(posterior_gammas, 0)
    
    post_prob_sum_iter10[rep,:]= marg_prob_sum

# torch.save(post_prob_sum_iter10, 'post_prob_sum_LiZhang_beta21_indepX_p1000_n100_w7_nu03_iter10.pt')


