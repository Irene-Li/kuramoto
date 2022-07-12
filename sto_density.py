import numpy as np 
import time 

class density_field_model: 

	def __init__(self, mu=None, nu=None, kappa=None, n=None, tau=None, epsilon=None):
		self.mu = mu 
		self.nu = nu 
		self.kappa = kappa 
		self.n = n 
		self.tau = tau 
		self.te = epsilon + tau 

	def initialise(self, T, dt, n_batches, psi0, noise_amp):
		self.T = T 
		self.dt = dt 
		self.n_batches = n_batches
		self.L = 1 
		self.X = int(self.te/self.dt)
		self.d2 = 0 
		self.d1 = int(self.X - int(self.tau/self.dt))
		self.psi0 = psi0 + np.random.normal(size=self.X)*noise_amp
		self.noise_amp = noise_amp

	def evolve(self): 
		start_time = time.time() 
		self.psi = np.zeros((self.n_batches, self.L))
		y = np.copy(self.psi0) 

		batch_size = int(self.T/self.dt)/self.n_batches 
		n = 0 
		for i in range(int(self.T/self.dt)): 
			newy = y[-1] + self._rhs(y)*self.dt + self._noise()

			y[0:-1] = y[1:]
			y[-1] = newy 
			if i  % batch_size == 0:
				self.psi[n] = y[-1] 
				n += 1 

		end_time = time.time() 
		print('time taken: ', end_time-start_time)


	def _hill_function(self, psi): 
		x = psi**self.n
		return x/(1+x)

	def _noise(self): 
		return np.random.normal()*np.sqrt(self.dt)*self.noise_amp

	def _rhs(self, y): 
		pos_feedback = self.mu*self._hill_function(y[self.d1])
		neg_feedback = self.nu*(1-self._hill_function(y[self.d2]))
		rhs = pos_feedback + neg_feedback - self.kappa*y[-1]

		return rhs 


class density_field_1D(density_field_model): 

	def __init__(self, mu=None, nu=None, kappa=None, n=None, tau=None, epsilon=None, tau_sigma=None, epsilon_sigma=None, c1=None, c2=None): 
		super().__init__(mu, nu, kappa, n, tau, epsilon)
		self.epsilon = epsilon 
		self.tau_sigma = tau_sigma
		self.epsilon_sigma = epsilon_sigma 
		self.c1 = c1 
		self.c2 = c2

	def initialise(self, L, T, dt, n_batches, psi0, noise_amp):
		self.taus = np.random.normal(self.tau, self.tau_sigma, L) 
		self.t_es= np.random.normal(self.epsilon, self.epsilon_sigma, L) + self.taus  

		self.T = T 
		self.n_batches = n_batches
		self.L = L 
		self.dt = dt 

		# find out the min and max delay and then how much past-time data we need to store 
		max_delay = max(self.t_es) 
		self.X = int(max_delay/self.dt)
		self.d2s = (self.X - self.t_es/self.dt).reshape((1, self.L)).astype('int')
		self.d1s = (self.X - self.taus/self.dt).reshape((1, self.L)).astype('int')
		self.psi0 = psi0 + np.random.normal(size=(self.X, self.L))*noise_amp
		self.noise_amp = noise_amp


	def _noise(self): 
		return np.random.normal(size=(self.L))*np.sqrt(self.dt)*self.noise_amp

	def _rhs(self, y): 

		past_psi1 = np.take_along_axis(y, self.d1s, axis=0)[0]
		past_psi2 = np.take_along_axis(y, self.d2s, axis=0)[0]

		h1 = self._hill_function(past_psi1)
		h2 = self._hill_function(past_psi2)

		pos_feedback = self.mu*(h1 + self.c1*(np.roll(h1, 1)+np.roll(h1, -1)))
		neg_feedback = self.nu*((1+self.c2*2) - h2 - self.c2*(np.roll(h2, 1)+np.roll(h2,-1))) # so the concentration does not go negative


		rhs = pos_feedback + neg_feedback - self.kappa*y[-1]

		return rhs 




class aging_model:

	def __init__(self, mu=None, nu=None, kappa=None, n=None, tau=None):
		self.mu = mu 
		self.nu = nu 
		self.kappa = kappa 
		self.n = n 
		self.tau = tau 

	def initialise(self, T, dt, n_batches, sigma):
		self.T = T 
		self.dt = dt 
		self.n_batches = n_batches
		self.X = int(self.tau*4/self.dt)
		self.d1 = int(self.tau*2/self.dt)
		self.d2 = int(self.tau*2.2/self.dt)
		self.d3 = int(self.tau*2.4/self.dt)
		self.d4 = int(self.tau*2.6/self.dt)
		self.sigma = sigma 

	def evolve(self): 
		start_time = time.time() 
		self.psi = np.zeros((self.n_batches, self.X))
		y = np.zeros(self.X)
		f = 1

		batch_size = int(self.T/self.dt)/self.n_batches 
		n = 0 
		for i in range(int(self.T/self.dt)): 
			newy = self._production(f) + self._noise()
			y[0] = newy 
			y[1:] = y[0:-1] 
			f += self._rhs(y, f)
			if i  % batch_size == 0:
				self.psi[n] = y 
				n += 1 

		end_time = time.time() 
		print('time taken: ', end_time-start_time)


	def _hill_function(self, psi): 
		x = psi**self.n
		return x/(1+x)

	def _production(self, f): 
		return 2*self._hill_function(f)

	def _rhs(self, y, f): 
		pos_feedback = self.mu*np.sum(self._hill_function(y[self.d1:self.d2]))*self.dt 
		neg_feedback = self.nu*np.sum(1-self._hill_function(y[self.d3:self.d4]))*self.dt
		return (pos_feedback + neg_feedback - self.kappa*f)*self.dt 

	def _noise(self): 
		noise = np.random.lognormal()*np.sqrt(self.dt)*self.sigma 
		return noise 



