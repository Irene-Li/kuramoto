
import numpy as np
from scipy.integrate import ode 
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from tqdm import tqdm

class Kuramoto():
    
    def __init__(self, epsilon, gamma, sigma, mean_omega, BC="PBC", grad=None):
        # Initialises the class with the model parameters 
        self.epsilon = epsilon 
        self.gamma = gamma 
        self.sigma = sigma
        self.mean_omega = mean_omega 
        self.BC = BC
        self._coupling = self._coupling1
        if BC == 'grad': 
            self.grad = grad 
        
    def initialise(self, L, T, n_frames, init=None, seed=None): 
        # Set up the simulation parameters 
        self.L = int(L) 
        self.size = int(L)
        self.T = T 
        self.n_frames = int(n_frames)
        self.step_size = T/float(n_frames)
        if seed is None:
            self.omegas = self.sigma*np.random.normal(size=(L)) + self.mean_omega
        else: 
            rs = RandomState(MT19937(SeedSequence(seed)))
            self.omegas = self.sigma*rs.normal(size=(L)) + self.mean_omega 
        if init is not None: 
            self.initial_state = init 
        else: 
            self.initial_state = np.zeros((self.L))

    def evolve(self):
        # The core function that integrates the ODEs forward. 
        
        self.res = np.zeros((self.n_frames, self.size))
        theta = np.copy(self.initial_state) 
        n = 0 

        f = lambda t, x: self._det_rhs(x)
        r = ode(f).set_integrator('lsoda', rtol=1e-6, first_step=1e-4)
        r.set_initial_value(theta, 0)

        for i in tqdm(range(self.n_frames)):
            if r.successful():
                self.res[i] = theta
                theta = r.integrate(r.t+self.step_size)
                theta = theta % (2*np.pi)
        
    def _coupling1(self, theta): 
        return np.sin(theta) + self.gamma*(1-np.cos(theta))

    def _coupling2(self, theta): 
        return theta + self.gamma*theta**2/2

    def _apply_bc(self, rhs, theta): 
        if self.BC == "fixed": 
            rhs[0] = 0 
            rhs[-1] = 0 
        if self.BC == "grad": 
            rhs[0] = self.omegas[0] + self.epsilon*(self._coupling(-self.grad[0])+self._coupling(theta[1]-theta[0]))
            rhs[-1] = self.omegas[-1] +self.epsilon*(self._coupling(self.grad[1])+self._coupling(theta[-2]-theta[-1]))
        return rhs 

    def _det_rhs(self, theta): 
        d_theta_1 = np.roll(theta, 1) - theta 
        d_theta_2 = np.roll(theta, -1) - theta 
        rhs = self.epsilon*(self._coupling(d_theta_1)+self._coupling(d_theta_2))+self.omegas
        rhs = self._apply_bc(rhs, theta) 
        return rhs 

class KuramotoNNN(Kuramoto): 

    def __init__(self, epsilon, gamma, sigma, mean_omega, alpha=1, BC="PBC", grad=None):
        if BC != 'grad': 
            print('unsupported BC')
        super().__init__(epsilon, gamma, sigma, mean_omega, BC, grad)
        self.alpha = alpha

    def _det_rhs(self, theta): 
        d_theta = theta[1:]-theta[:-1]
        d_theta = np.concatenate([[self.grad[0]], d_theta, [self.grad[1]]])
        d_theta2 = d_theta[1:]+d_theta[:-1]
        d_theta2 = np.concatenate([[2*self.grad[0]], d_theta2, [2*self.grad[1]]])
        
        nn = self._coupling(d_theta[1:]) + self._coupling(-d_theta[:-1])
        nnn = self._coupling(d_theta2[2:]) + self._coupling(-d_theta2[:-2]) 
        rhs = self.epsilon*(nn + self.alpha*nnn)+self.omegas
        return rhs 
    
class KuramotoVF(Kuramoto): 
    
    def __init__(self, epsilon, gamma, sigma, mean_omega, noise=0.01, BC="PBC", grad=None):
        super().__init__(epsilon, gamma, sigma, mean_omega, BC, grad)
        self.noise = noise 
    
    def evolve(self, dt=1e-2):
        # The core function that integrates the ODEs forward. 
        
        self.res = np.zeros((self.n_frames, self.size))
        theta = np.copy(self.initial_state) 

        f = lambda t, x: self._rhs(x)
        for i in tqdm(range(self.n_frames)):
            for j in range(int(self.step_size/dt)): 
                theta += self._rhs(theta, dt)                
            theta = theta % (2*np.pi) 
            self.res[i] = theta
                
    def _rhs(self, theta, dt): 
        d_theta_1 = np.roll(theta, 1) - theta 
        d_theta_2 = np.roll(theta, -1) - theta 
    
        rhs = self.epsilon*(self._coupling(d_theta_1)+self._coupling(d_theta_2))+self.omegas
        rhs = self._apply_bc(rhs, theta) 
        rhs *= dt 
        rhs += np.sqrt(dt)*self.noise*np.random.normal(size=(self.L))
        return rhs 

        
class KuramotoNetwork(Kuramoto): 

    def initialise(self, L, T, n_frames, network_matrix, seed=None): 
        super().initialise(L, T, n_frames, seed=None)
        self.M = np.copy(network_matrix)
        self.M += np.eye(L, k=1) + np.eye(L, k=-1)

    def _det_rhs(self, theta): 
        rhs = np.copy(self.omegas)
        for i in range(self.L): 
            for j in range(self.L): 
                if self.M[i, j] > 0:
                    d_theta = theta[j] -  theta[i]
                    rhs[i] += self.epsilon*(self._coupling(d_theta))
        return rhs 
    
class Kuramoto2D(Kuramoto): 
    
    def initialise(self, Lx, Ly, T, n_frames, seed=None, init=None): 
        self.Lx = int(Lx)
        self.Ly = int(Ly)
        self.size = self.Lx*self.Ly 
        self.T = T 
        self.n_frames = int(n_frames)
        self.step_size = T/float(n_frames)

        if seed is None:
            self.omegas = self.sigma*np.random.normal(size=(Lx, Ly)) + self.mean_omega
        else: 
            rs = RandomState(MT19937(SeedSequence(seed)))
            self.omegas = self.sigma*rs.normal(size=(Lx, Ly)) + self.mean_omega 
            
        if init is not None: 
            self.initial_state = init 
        else: 
            self.initial_state = np.zeros((Lx*Ly))
             
    def _det_rhs(self, theta): 
        theta = theta.reshape((self.Lx, self.Ly))
        d_thetas = [] 
        for d in [1, -1]:
            for a in [0, 1]: 
                d_thetas.append((np.roll(theta, d, axis=a) - theta))
                
        coupling = sum(map(self._coupling, d_thetas))
        rhs = self.epsilon*coupling + self.omegas
        if self.BC == "fixed": 
            rhs[0] = 0 
            rhs[-1] = 0
        if self.BC == "grad": 
            rhs[0] = self.omegas[0] + self.epsilon*(self._coupling(-self.grad[0])+self._coupling(theta[1]-theta[0]))
            rhs[-1] = self.omegas[-1] +self.epsilon*(self._coupling(self.grad[1])+self._coupling(theta[-2]-theta[-1]))   
        return rhs.flatten()