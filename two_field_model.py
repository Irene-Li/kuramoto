import numpy as np 
from jitcdde import jitcdde, y, t
import time 

class two_field_model: 

	def __init__(self, mu=None, nu=None, gamma=None, kappa1=None, kappa2=None, n=None, tau=None, epsilon=None):
		self.mu = mu 
		self.nu = nu 
		self.gamma = gamma 
		self.kappa1 = kappa1 
		self.kappa2 = kappa2 
		self.n = n 
		self.tau = tau 
		self.epsilon = epsilon 

	def initialise(self, T, n_batches, psi0, f0):
		self.T = T 
		self.n_batches = n_batches
		self.X = 1 
		self.taus = [self.tau]
		self.t_es = [self.tau + self.epsilon]
		self.y0 = np.zeros((2*self.X))
		self.y0[:self.X] = psi0 
		self.y0[self.X:] = f0 

	def evolve(self): 
		f = self._rhs(y, t)

		print('Setting up integrator...')
		start_time = time.time() 
		dde = jitcdde(f, n=2*self.X, verbose=False, max_delay=max(self.t_es))
		dde.constant_past(self.y0)
		dde.step_on_discontinuities()
		dde.set_integration_parameters(atol=1e-5)
		end_time = time.time() 
		print('time taken: ', end_time-start_time)


		print('Performing integration...')
		start_time = time.time() 
		self.ys = np.zeros((self.n_batches, 2*self.X))
		self.ts = np.zeros((self.n_batches))
		for (i, tt) in enumerate(np.linspace(dde.t, dde.t+self.T, self.n_batches)):
			self.ts[i] = tt
			self.ys[i] = dde.integrate(tt)
		end_time = time.time() 
		print('time taken: ', end_time-start_time)


	def _hill_function(self, psi): 
		x = psi**self.n
		return x/(1+x)


	def _rhs(self, y, t): 
		pos_feedback = self.mu*self._hill_function(y(1, t-self.tau))
		neg_feedback = self.nu*(1-self._hill_function(y(1, t-self.tau-self.epsilon)))
		production = self.gamma*self._hill_function(y(0))

		return [pos_feedback + neg_feedback - self.kappa1*y(0),  production - self.kappa2*y(1)]