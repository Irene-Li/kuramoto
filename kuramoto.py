
import numpy as np
from scipy.integrate import ode 
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

class Kuramoto():
    
    def __init__(self, epsilon, gamma, sigma, mean_omega, BC="fixed"):
        # Initialises the class with the model parameters 
        self.epsilon = epsilon 
        self.gamma = gamma 
        self.sigma = sigma
        self.mean_omega = mean_omega 
        self.BC = BC
        
    def initialise(self, L, T, dt, n_batches, seed=None): 
        # Set up the simulation parameters 
        self.L = int(L) 
        self.size = int(L)
        self.T = T 
        self.dt = dt 
        self.n_batches = int(n_batches)
        self.step_size = T/(self.n_batches-1)
        self.batch_size = int(np.floor(self.step_size/self.dt))
        if seed is None:
            self.omegas = self.sigma*np.random.normal(size=(L)) + self.mean_omega
        else: 
            rs = RandomState(MT19937(SeedSequence(seed)))
            self.omegas = self.sigma*rs.normal(size=(L)) + self.mean_omega 

    def evolve(self, verbose=False):
        # The core function that integrates the ODEs forward. 
        
        self.res = np.zeros((self.n_batches, self.size))
        theta = np.zeros((self.size))
        n = 0 
        
        small_batch = self.batch_size
        while small_batch > 1000:
            small_batch /= 10 # decrease the amount of time integration at each step
        
        f = lambda t, x: self._det_rhs(x)
        r = ode(f).set_integrator('lsoda', rtol=1e-5, nsteps=small_batch)
        r.set_initial_value(theta, 0)

        for i in range(int((self.T/self.dt)/small_batch)):
            if r.successful():
                if i % int(self.batch_size/small_batch) == 0:
                    self.res[n] = theta
                    if verbose: 
                        print("time step: {} \n".format(n))
                    n += 1
                theta = r.integrate(r.t+self.dt*small_batch)
        
    def _coupling(self, theta): 
        return np.sin(theta) + self.gamma*(1-np.cos(theta))

    def _apply_bc(self, rhs): 
        if self.BC == "fixed": 
            rhs[0] = 0 
            rhs[-1] = 0
        if self.BC == "open": 
            rhs[0] = self.omegas[0]
            rhs[-1] = self.omegas[-1]
        return rhs 

    def _det_rhs(self, theta): 
        d_theta_1 = (np.roll(theta, 1) - theta ) % (2*np.pi) 
        d_theta_2 = (np.roll(theta, -1) - theta ) % (2*np.pi)
        rhs = self.epsilon*(self._coupling(d_theta_1)+self._coupling(d_theta_2))+self.omegas
        rhs = self._apply_bc(rhs) 
        return rhs 

class KuramotoNetwork(Kuramoto): 

    def initialise(self, L, T, dt, n_batches, network_matrix, seed=None): 
        super().initialise(L, T, dt, n_batches, seed=None)
        self.M = np.copy(network_matrix)
        self.M += np.eye(L, k=1) + np.eye(L, k=-1)

    def _det_rhs(self, theta): 
        rhs = np.copy(self.omegas)
        for i in range(self.L): 
            for j in range(self.L): 
                if self.M[i, j] > 0:
                    d_theta = theta[j] -  theta[i]
                    rhs[i] += self.epsilon*(self._coupling(d_theta))
        rhs = self._apply_bc(rhs) 
        return rhs 
    
class Kuramoto2D(Kuramoto): 
    
    def initialise(self, Lx, Ly, T, dt, n_batches, seed=None): 
        self.Lx = int(Lx)
        self.Ly = int(Ly)
        self.size = self.Lx*self.Ly 
        self.T = T 
        self.dt = dt 
        self.n_batches = int(n_batches)
        self.step_size = T/(self.n_batches-1)
        self.batch_size = int(np.floor(self.step_size/self.dt))

        if seed is None:
            self.omegas = self.sigma*np.random.normal(size=(Lx, Ly)) + self.mean_omega
        else: 
            rs = RandomState(MT19937(SeedSequence(seed)))
            self.omegas = self.sigma*rs.normal(size=(Lx, Ly)) + self.mean_omega 
             
    def _det_rhs(self, theta): 
        theta = theta.reshape((self.Lx, self.Ly))
        d_thetas = [] 
        for d in [1, -1]:
            for a in [0, 1]: 
                d_thetas.append((np.roll(theta, d, axis=a) - theta))
                
        coupling = sum(map(self._coupling, d_thetas))
        rhs = self.epsilon*coupling + self.omegas
        if self.BC == "fixed": 
            rhs[0, :] = 0 
            rhs[-1, :] = 0
        if self.BC == "open": 
            rhs[0] = self.omegas[0]
            rhs[-1] = self.omegas[-1]        
        return rhs.flatten()