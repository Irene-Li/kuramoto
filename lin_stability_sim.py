import numpy as np
from matplotlib import pyplot as plt
from sto_density import * 


kappa = 1
n = 6
mu = 1.2
nu = 1.2


T = 4e3
dt = 1e-2
sigma = 0
n_batches = int(T)


epsilons = np.arange(0, 8, 0.1)
taus = np.arange(0.1, 25, 0.1)
amps = [] 
for (i, epsilon) in enumerate(epsilons):
    for (j, tau) in enumerate(taus): 
        m = density_field_model(mu, nu, kappa, n, tau, epsilon)
        m.initialise(T, dt, n_batches, 1, sigma)
        m.evolve() 
        
        
        y = np.copy(m.psi[-200:]).flatten() 
        yfft = np.fft.rfft(y)
        max_amp = max(np.abs(yfft[1:]))
        amps.append((epsilon, tau, max_amp))



amps = np.array(amps) 
np.save('sim_data/amps_mu{}_nu{}.npy'.format(mu, nu), amps)