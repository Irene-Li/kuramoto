import numpy as np
from scipy.optimize import root_scalar, minimize

class model:

	def __init__(self, mu, nu, kappa, n, tau, epsilon):
		self.mu = mu 
		self.nu = nu 
		self.kappa = kappa 
		self.n = n 		
		self.tau = tau 
		self.epsilon = epsilon 

	def set_mu_nu(self, mu, nu):
		self.mu = mu 
		self.nu = nu  

	def f(self, phi): 
		pos_feedback = self.mu*self.hill_function(phi)
		neg_feedback = self.nu*(1-self.hill_function(phi))

		return pos_feedback + neg_feedback - self.kappa*phi

	def hill_function(self, phi): 
		x = phi**self.n
		return x/(1+x)

	def hill_prime(self, phi): 
		return self.n*(phi**(self.n-1))/((phi**self.n +1)**2)

	def rhs(self, lbda): 
		return self.mu_bar*np.exp(-lbda*self.tau) - self.nu_bar*np.exp(-lbda*(self.tau+self.epsilon)) - self.kappa 

	def find_phi_bar(self, max_val): 
		phi_bar = root_scalar(self.f, bracket=(0, max_val)).root
		self.phi_bar = phi_bar 
		hp = self.hill_prime(phi_bar)
		self.mu_bar = hp*self.mu 
		self.nu_bar = hp*self.nu 
		return phi_bar 

	def get_e_star(self): 
		omega_star = np.sqrt(self.nu_bar**2-(self.kappa-self.mu_bar)**2)
		return np.angle(-self.kappa+self.nu_bar+1j*omega_star)/omega_star

	def get_tau_star(self): 
		omega_star = np.sqrt((self.nu_bar-self.mu_bar)**2 - self.kappa**2)
		return np.angle(kappa-1j*omega_star)/omega_star


	def solve_for_omega(self, epsilon):
		self.epsilon = epsilon 
		f = lambda x: self._y1(x)-self._y2(x)
		a = 0 
		b = 2*np.pi/epsilon
		v1 = f(a) 
		v2 = f(b) 
		if v1*v2 < 0: 
			return [root_scalar(f, bracket=(a, b)).root]
		else:
			if v1 < 0: 
				print('both values smaller than zero')
			else:
				m = minimize(f, a+1e-3, bounds=[(a, b)]).x[0]
				if f(m) > 0: 
					return None # no root 
				else:
					r1 = root_scalar(f, bracket=(a, m)).root
					r2 = root_scalar(f, bracket=(m, b)).root 
					return [r1, r2]  

	def solve_for_tau(self, omega):
		beta = np.angle(self.kappa - 1j*omega)
		alpha = np.angle(self.mu - self.nu*np.exp(1j*omega*self.epsilon))
		tau = ((beta - alpha) % (2*np.pi))/omega
		return tau, tau+2*np.pi/omega, tau+4*np.pi/omega

	def obtain_grads(self, omega, tau): 
		self.tau = tau 
		A = omega*omega - 1j*omega*self.kappa
		B = 1j*omega*self.nu_bar*np.exp(-1j*omega*(self.tau+self.epsilon))
		C = 1 + self.tau*(self.kappa+1j*omega) - self.epsilon*self.nu_bar*np.exp(-1j*omega*(self.tau+self.epsilon))
		return np.real(A/C), np.real(B/C)

	def _y1(self, omega): 
		return omega**2 + self.kappa**2 

	def _y2(self, omega): 
		return self.mu_bar**2 + self.nu_bar**2 - 2*self.mu_bar*self.nu_bar*np.cos(omega*self.epsilon)


