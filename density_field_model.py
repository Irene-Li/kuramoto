import numpy as np 
from jitcdde import jitcdde, y, t
import time 

class density_field_model: 

	def __init__(self, alpha=None, mu=None, nu=None, kappa=None, n=None, tau=None, epsilon=None):
		self.alpha = alpha 
		self.mu = mu 
		self.nu = nu 
		self.kappa = kappa 
		self.n = n 
		self.tau = tau 
		self.epsilon = epsilon 

	def initialise(self, T, n_batches, psi0):
		self.T = T 
		self.n_batches = n_batches
		self.X = 1 
		self.taus = [self.tau]
		self.t_es = [self.tau + self.epsilon]
		self.psi0 = psi0 

	def evolve(self): 
		f = self._rhs(y, t)

		print('Setting up integrator...')
		start_time = time.time() 
		dde = jitcdde(f, n=self.X, verbose=False, max_delay=max(self.t_es))
		dde.constant_past(np.zeros((self.X))+self.psi0)
		dde.step_on_discontinuities()
		dde.set_integration_parameters(atol=1e-5)
		end_time = time.time() 
		print('time taken: ', end_time-start_time)


		print('Performing integration...')
		start_time = time.time() 
		self.psi = np.zeros((self.n_batches, self.X))
		self.ts = np.zeros((self.n_batches))
		for (i, tt) in enumerate(np.linspace(dde.t, dde.t+self.T, self.n_batches)):
			self.ts[i] = tt
			self.psi[i] = dde.integrate(tt)
		end_time = time.time() 
		print('time taken: ', end_time-start_time)


	def _hill_function(self, psi): 
		x = psi**self.n
		return x/(1+x)


	def _rhs(self, psi, t): 
		pos_feedback = self.mu*self._hill_function(psi(0, t-self.tau))
		neg_feedback = self.nu*self._hill_function(psi(0, t-self.tau-self.epsilon))

		return [self.alpha + pos_feedback - neg_feedback - self.kappa*psi(0)]

class density_field_model_1D(density_field_model): 

	def __init__(self, alpha=None, mu=None, nu=None, kappa=None, n=None, tau=None, epsilon=None, tau_sigma=None, epsilon_sigma=None, c1=None, c2=None): 
		super().__init__(alpha, mu, nu, kappa, n, tau, epsilon)
		self.tau_sigma = tau_sigma
		self.epsilon_sigma = epsilon_sigma 
		self.c1 = c1 
		self.c2 = c2

	def initialise(self, X, T, n_batches):
		self.taus = np.random.normal(self.tau, self.tau_sigma, X) 
		self.t_es= np.random.normal(self.epsilon, self.epsilon_sigma, X) + self.taus
		self.X = X 
		self.T = T 
		self.n_batches = n_batches
		self._nearest_neighbour_matrix()

	def _rhs(self, psi, t): 

		f = [] 

		for i in range(self.X):
			h1 = self._hill_function(psi(i, t-self.taus[i])) 
			h2 = self._hill_function(psi(i, t-self.t_es[i]))

			# nns = self._find_nearest_neighbours(i)
			# h1 += self.c1*sum(self._hill_function(psi(j, t-self.taus[j])) for j in nns)
			# h2 += self.c2*sum(self._hill_function(psi(j, t-self.t_es[j])) for j in nns)


			h1 += self.c1*sum(self._hill_function(psi(j, t-self.taus[j])) for j in range(self.X) if self.M[i,j])
			h2 += self.c2*sum(self._hill_function(psi(j, t-self.t_es[j])) for j in range(self.X) if self.M[i,j])

			rhs = self.alpha + self.mu*h1 - (self.nu*h2 + self.kappa)*psi(i)
			f.append(rhs)

		return f 

	def _nearest_neighbour_matrix(self): 
		self.M = np.zeros((self.X, self.X))
		for i in range(self.X): 
			nns = self._find_nearest_neighbours(i)
			for j in nns: 
				self.M[i, j] = 1 

	def _find_nearest_neighbours(self, index):
		return [(index-1)%self.X, (index+1)%self.X]

class density_field_model_2D(density_field_model_1D):

	def initialise(self, Lx, Ly, T, n_batches):
		self.Lx = Lx 
		self.Ly = Ly 
		super().initialise(Lx*Ly, T, n_batches)

	def _find_nearest_neighbours(self, index): 
		i = int(index/self.Ly)
		j = index % self.Ly 
		indices = [] 
		for a in [(i-1) % self.Lx, (i+1) % self.Lx]:
			indices.append(a*self.Ly+j)
		for b in [(j-1) % self.Ly, (j+1) % self.Ly]:
			indices.append(i*self.Ly+b) 
		return indices 







