import numpy as np
from scipy.optimize import root_scalar, minimize, brute
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt 


class Inference(): 

	def __init__(self): 
		pass 

	def MAP(self, phi):
		self.phi = phi 
		self.L = len(phi)+1
	    self.res = brute(self.cost, ((-1, 1), (-np.pi, np.pi)))
	    return res.x, res.fun 


	def cost(self, eta, grad): 
	    BB, d = self.find_BB(eta, grad)
	    _, b = self.traj_logpdf(BB)
	    ll = b + np.log(d)
	    return -np.sum(ll)


	def find_BB(self, eta, grad): 
	    '''
	    Find the Brownian Bridge from phi for fixed values of eta and gradients at the boundary 
	    '''
	    m_grad = 2*np.sin(grad)/self.L 
	    m_eta = (2*np.sum((1-np.cos(self.phi))) + 2*(1-np.cos(grad)))/self.L
	    x = np.arange(1, self.L)
	    BB = np.sin(phi) - np.sin(-grad) + eta*(1-np.cos(grad)) + eta*(1-np.cos(self.phi))- m_grad*x - eta*m_eta*x 
	    BB[1:] += 2*eta*np.cumsum(1-np.cos(self.phi[:-1]))        
	    J = np.zeros((self.L-1, self.L-1))
	    diag = np.cos(self.phi) + eta*np.sin(self.phi)
	    np.fill_diagonal(J, diag)
	    J -= eta*2*np.outer(x, np.sin(self.phi))/self.L  
	    for j in range(self.L-1): 
	        J[j+1:, j] += 2*eta*np.sin(self.phi[j])

	    return BB, np.abs(np.linalg.det(J))


	def get_cov(self): 
		N = self.L - 1 
	    M = np.tril(np.ones((N, N)))[:-1, :] - (np.arange(1, N)/N)[:, np.newaxis]
	    Cov = M @ np.identity(N) @ M.T
	    return Cov 


	def traj_logpdf(self, BB): 
	    Cov = self.get_cov(len(BB)+1)
	    invCov = np.linalg.inv(Cov)
	    var = BB.T @ invCov @ BB
	    var /= len(BB)
	    return var, stats.multivariate_normal.logpdf(BB, cov=Cov*var)


	def cost_simple(self, eta, sigma, grad, Cov): 
	    BB, d = find_BB(eta, grad)
	    logpdf = stats.multivariate_normal.logpdf(BB, cov=Cov*sigma*sigma)
	    ll = logpdf + np.log(d)
	    return -np.sum(ll)
	    

	def get_MAP_sigma(self): 
		eta, grad = self.res.x 
	    BB, _ = find_BB(self.phi, eta, grad)
	    var, _ = traj_logpdf(BB)
	    self.sigma = np.sqrt(var)
	    return self.sigma 

	def plot_cost(self, widths=None, N=100):
	    Cov = get_cov()
	    eta_map, grad_map = self.res.x 
	    sigma_map = self.sigma 
	    
	    cost_map = cost_simple(eta_map, sigma_map, grad_map, Cov) 
	    CI = cost_map+1.92
	    
	    if widths is None: 
	        widths = [1, 0.1*sigma_map, np.pi]
	    
	    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(21, 6))
	    
	    x = [eta_map, sigma_map, grad_map]
	    for i in range(3): 
	        params = np.linspace(x[i]-widths[i], x[i]+widths[i], N)
	        c= [] 
	        x_copy = np.copy(x)
	        for p in params: 
	            x_copy[i] = p 
	            c.append(cost_simple(*x_copy, Cov))
	        axes[i].plot(params, c)
	        axes[i].axhline(y=CI)
	        axes[i].set_ylim([cost_map-2, cost_map+10])
	    plt.show()

	    
	def errors_hess(self, diff=1e-4):
	    Cov = self.get_cov()
	    cost_map = self.res.fun 
	    eta, grad = self.res.x 
	    x = [eta, self.sigma, grad] 
	    err = [] 
	    for i in range(3): 
	        x[i] += diff
	        a = self.cost_simple(*x, Cov) 
	        x[i] -= diff*2 
	        b = self.cost_simple(*x, Cov) 
	        x[i] += diff 
	        hess = 2*(a+b-2*cost_map)/(diff**2) # Curvature 
	        err.append(1/np.sqrt(hess))
	    return err 

	def errors_brute(self, diff=None, N=500):
	    Cov = self.get_cov()
	    cost_map = self.res.fun 
	    eta, grad = self.res.x 
	    x = [eta, self.sigma, grad] 
	    err = [] 
	    
	    if diff is None: 
	        diff = [1, 0.1*sigma, np.pi]
	    for i in range(3): 
	        param = np.linspace(x[i]-diff[i], x[i]+diff[i], N)
	        c = [] 
	        x_copy = np.copy(x)
	        for p in param:
	            x_copy[i] = p 
	            c.append(cost_simple(phi, *x_copy, Cov))
	        c = np.array(c) 
	        mask = (c<(cost_map+1.92)).astype('int') 
	        d = mask[1:] - mask[:-1]
	        index1 = np.argwhere(d>0)[0,0]
	        index2 = np.argwhere(d<0)[-1,0]
	        if mask[0] > 0: 
	            index1 = 0 
	        if mask[-1] > 0: 
	            index2 = N 
	        err.append((index2-index1)*(2*diff[i]/N)/2)
	    return err 
	    



