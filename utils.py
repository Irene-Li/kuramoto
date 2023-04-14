import numpy as np
from scipy.optimize import root_scalar, minimize, brute
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt 

# ==== process phase information === 

def smooth(f, width): 
    '''
    Average f over width. 
    The output is of length len(f)/width
    '''
    N = len(f)
    length = int(np.floor(N/width))
    f = f[:length*width]
    f_smooth = np.mean(f.reshape((length, width)), -1)
    return f_smooth 

def shift(phases, tol=1): 
    '''
    Shift the phases such that all data points are continuous 
    '''
    z = np.exp(phases*1j) 
    diff_z = np.abs(z[1:]-z[:-1])
    diff_ang = phases[1:]-phases[:-1]
    for (i, (dz, dtheta)) in enumerate(zip(diff_z, diff_ang)): 
        if np.abs(dtheta) - dz > tol:  
            if dtheta < 0:
                phases[i+1:] += np.pi*2
            else: 
                phases[i+1:] -= np.pi*2 

# == convert between phases and stages =============

# find durations of each stage from [Oakberg 1956] to convert stage to phase 
durations = np.array([22, 18.1, 8.7, 18.6, 11.3, 18.1, 20.6, 20.8, 15.2, 11.3, 21.4, 20.8])
boundaries = (np.cumsum(durations))/np.sum(durations)*np.pi*2
midpoints = (np.cumsum(durations)-durations/2)/np.sum(durations)*np.pi*2

def stage_to_phase(stage): 
    return midpoints[int(stage)-1]

def phase_to_stage(phase): 
    return np.searchsorted(boundaries, phase)+1

def is_invalid(x): 
    return (x < 1) | (x > 12 ) 

def period_average(x, y): 
    z1 = np.exp(1j*stage_to_phase(x))
    z2 = np.exp(1j*stage_to_phase(y))
    theta = np.angle(z1+z2)
    if theta < 0: 
        theta += 2*np.pi 
    return phase_to_stage(theta)


# ========== Maximum Likelihood Estimation ==================

def find_BB(phi, eta, grad): 
    '''
    Find the Brownian Bridge from phi for fixed values of eta and gradients at the boundary 
    '''
    L = len(phi)+1
    m_grad = 2*np.sin(grad)/L 
    m_eta = (2*np.sum((1-np.cos(phi))) + 2*(1-np.cos(grad)))/L
    x = np.arange(1, L)
    BB = np.sin(phi) - np.sin(-grad) + eta*(1-np.cos(grad)) + eta*(1-np.cos(phi))- m_grad*x - eta*m_eta*x 
    BB[1:] += 2*eta*np.cumsum(1-np.cos(phi[:-1]))        
    J = np.zeros((L-1, L-1))
    diag = np.cos(phi) + eta*np.sin(phi)
    np.fill_diagonal(J, diag)
    J -= eta*2*np.outer(x, np.sin(phi))/L  
    for j in range(L-1): 
        J[j+1:, j] += 2*eta*np.sin(phi[j])

    return BB, np.abs(np.linalg.det(J))


def exact_BB(omegas): 
    BBs = [] 
    for omega in omegas: 
        L = len(omega)
        x = np.arange(1, L+1)
        W = np.cumsum(omega)
        BB = (W - W[-1]*x/L)[:-1]
        BBs.append(-BB)
    return BBs 


def get_cov(N): 
    M = np.tril(np.ones((N, N)))[:-1, :] - (np.arange(1, N)/N)[:, np.newaxis]
    Cov = M @ np.identity(N) @ M.T
    return Cov 


def traj_logpdf(BB): 
    Cov = get_cov(len(BB)+1)
    invCov = np.linalg.inv(Cov)
    var = BB.T @ invCov @ BB
    var /= len(BB)
    return var, stats.multivariate_normal.logpdf(BB, cov=Cov*var)

def cost(phi, eta, grad): 
    BB, d = find_BB(phi, eta, grad)
    _, b = traj_logpdf(BB)
    ll = b + np.log(d)
    return -np.sum(ll)
    
def MAP(phi): 
    f = lambda x: cost(phi, x[0], x[1]) 
    res = brute(f, ((-1, 1), (-np.pi, np.pi)))
    return res

def plot_cost(phi, g): 
    N = 50 
    etas = np.linspace(-2, 2, N)
    c= [] 
    for eta in etas: 
        c.append(cost(phi, eta, grad))
    plt.plot(etas, c)
    plt.title('cost')
    plt.show()