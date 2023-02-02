import numpy as np
from scipy.optimize import root_scalar, minimize
from scipy.signal import find_peaks
import pandas as pd
from scipy.stats import norm
from scipy.stats import gamma as gamma_dist
from matplotlib import pyplot as plt 


# ========================================

def find_defects(datasets): 
    for dataset in datasets:
        p = dataset['phase']
        l = dataset['length']
        diff =  (p[1:] - p[:-1]) % 3
        sign = diff[0] 
        dataset['diff'] = diff 
        dataset['defect'] = [] 
        for (i, d) in enumerate(diff): 
            if d != sign: 
                dataset['defect'].append(i+1)
                sign = d 

def find_lengths(datasets): 
    lengths = [[], [], []]
    lengths_wo_defects = [[], [], []]
    for dataset in datasets: 
        p = dataset['phase']
        l = dataset['length']
        d = dataset['defect']
        for i in range(3): 
            lengths[i].extend([x for (x, y) in zip(l, p) if y == i])
            lengths_wo_defects[i].extend([x for (a, (x, y)) in enumerate(zip(l, p))  if (y == i and a not in d)])

    return lengths, lengths_wo_defects

# ==========================================

def process_file(filename): 
    # first, read in the csv files
    dataframe = pd.read_csv(filename)
    
    # extract the columns we want and concatenate them together 
    sub_dataframe = concat_columns(dataframe)
    
    # make list of datasets from the sub_dataframe 
    datasets = make_datasets(sub_dataframe) 
    
    # filter out the segments that are too short 
    f = lambda x: len(x['length']) > 2 
    datasets = list(filter(f, datasets))
    
    return datasets 
    
def concat_columns(dataframe):
    temp = {} 
    for col in dataframe.columns: 
        full_name = col.split('.')
        if len(full_name) == 1: 
            temp[col] = dataframe[col]
        if len(full_name) == 2:
            temp[full_name[0]] = pd.concat([temp[full_name[0]], dataframe[col]], ignore_index=True)
    sub_dict = {k: temp[k] for k in ['Branch', 'Length', 'Phase']}
    sub_dataframe = pd.DataFrame(sub_dict)
    return sub_dataframe 
    
def make_datasets(sub_dataframe): 
    datasets = [] 
    flag = True
    phase_dict = {'6': 0, '8':1, '2':2 }
    for (i, row) in sub_dataframe.iterrows():
        if row.isnull().all():
            flag = True
        else: 
            if flag:                
                # start a new dataset 
                dataset = {} 
                datasets.append(dataset)
                dataset['ends'] = []
                dataset['length'] = [] 
                dataset['phase'] = [] 
                flag = False 
            try: 
                dataset['length'].append(int(row['Length'].replace(',', '')))
                dataset['phase'].append(phase_dict[row['Phase'][-1]])
            except Exception: 
                print('Some error occured: \n', row)
            if isinstance(row['Branch'], str): 
                dataset['ends'].append(row['Branch'].lower())
    for dataset in datasets: 
        dataset['length'] = np.array(dataset['length'])
        dataset['phase'] = np.array(dataset['phase'])
    return datasets  

def collect_dataset(phases, breakpoints): 
    indexed_phases = np.zeros_like(phases) 
    indexed_phases[phases > breakpoints[0]] = 1 
    indexed_phases[phases > breakpoints[1]] = 2 
    
    
    dataset = {'length': [], 'phase': []}
    w = 1
    q = indexed_phases[0]
    for p in indexed_phases[1:]: 
        if p == q: # if same phase as previous one 
            w += 1 
        else: 
            dataset['length'].append(w)
            dataset['phase'].append(q)
            w = 1 
        q = p 

    dataset['length'] = np.array(dataset['length'])
    dataset['phase'] = np.array(dataset['phase'])
    return dataset

def calculate_gradients(phases):
    z = np.exp(phases*1j) 
    dpdx = np.imag((z[2:]-z[:-2])/2/z[1:-1])
    return dpdx 

def smooth(phases, width): 
    z = np.exp(phases*1j)
    z_smooth = np.convolve(z, np.ones(width), mode='valid')/width
    theta_smooth = np.angle(z_smooth) % (2*np.pi)
    return theta_smooth 

def smooth2(phases, width): 
    z = np.exp(phases*1j)
    N = len(z)
    length = int(np.floor(N/width))
    z = z[:length*width]
    z_smooth = np.mean(z.reshape((length, width)), -1)
    theta_smooth = np.angle(z_smooth) % (2*np.pi)
    return theta_smooth 

def shift(phases, tol=1): 
    z = np.exp(phases*1j) 
    diff_z = np.abs(z[1:]-z[:-1])
    diff_ang = phases[1:]-phases[:-1]
    for (i, (dz, dtheta)) in enumerate(zip(diff_z, diff_ang)): 
        if np.abs(dtheta) - dz > tol:  
            if dtheta < 0:
                phases[i+1:] += np.pi*2
            else: 
                phases[i+1:] -= np.pi*2 

def find_turning_points(phases, prominence): 
    peaks, _ = find_peaks(phases, prominence=prominence)
    peaks_2, _ = find_peaks(-phases, prominence=prominence)
    return np.concatenate([peaks, peaks_2])


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


# ==================================================


def plot(freqs, gamma, sigma):     
    plt.hist(freqs, bins=10, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, 0, sigma)
    plt.plot(x, p, 'k', linewidth=2)
    plt.show() 
    
def Gamma(x, gamma): 
    return np.sin(x) + gamma*(1-np.cos(x))

def get_freqs(data, gamma):
    N = len(data)
    Omega = 2*gamma*np.sum(1 - np.cos(data[1:] - data[:-1]))/N
    freqs = Omega - Gamma(data[2:]-data[1:-1], gamma) - Gamma(data[:-2]-data[1:-1],gamma)
    return freqs 

def minus_log_likelihood(data, gamma):
    freqs = get_freqs(data, gamma)
    std = np.sqrt(np.sum(freqs**2)/len(freqs))
    return -np.sum(norm.logpdf(freqs, 0, std))

def minus_log_prior(gamma, alpha, beta):
    return -gamma_dist.logpdf(gamma, alpha, scale=1/beta)

def MAP_sigma(data, gamma): 
    freqs = get_freqs(data, gamma)
    return np.sqrt(np.sum(freqs**2)/len(freqs))

def MAP_gamma(data, alpha, beta): 
    f = lambda x: minus_log_likelihood(data, x) + minus_log_prior(x, alpha, beta)
    res = minimize(f, 0.1)
    return res.x[0]

def errors(data, map_gamma, map_sigma, alpha, beta):

    def f(gamma, sigma):
        freqs = get_freqs(data, gamma)
        a = -np.sum(norm.logpdf(freqs, 0, sigma))
        b = minus_log_prior(gamma, alpha, beta)
        return a + b 

    gamma_step = 1e-5
    sigma_step = 1e-5 
    MAP_val = f(map_gamma, map_sigma)
    
    test_gammas = [map_gamma+gamma_step, map_gamma, map_gamma-gamma_step]
    test_sigmas = [map_sigma+sigma_step, map_sigma, map_sigma-sigma_step]
    
    test_vals = np.reshape([f(x, y) for x in test_gammas for y in test_sigmas], (3, 3))
    hess = np.zeros((2, 2))
    hess[0, 0] = (test_vals[0, 1] + test_vals[2, 1] - 2*test_vals[1, 1])/(gamma_step*gamma_step)
    hess[1, 1] = (test_vals[1, 0] + test_vals[1, 2] - 2*test_vals[1, 1])/(sigma_step*sigma_step)
    hess[1, 0] = (test_vals[2, 2] + test_vals[0, 0] - test_vals[2, 0] - test_vals[0, 2])/(4*gamma_step*sigma_step)
    hess[0, 1] = hess[1, 0]
    cov_mat = np.linalg.inv(hess)
    return np.sqrt(cov_mat[0, 0]), np.sqrt(cov_mat[1, 1])


def plot_prior_and_posterior(data, alpha, beta):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))

    f = lambda x: minus_log_prior(x, alpha, beta)
    gammas = np.linspace(0, 1, 100)
    axes[0].plot(gammas, list(map(f, gammas)))
    axes[0].set_title('Prior')

    f = lambda x: minus_log_likelihood(data, x) + minus_log_prior(x, alpha, beta)
    gammas = np.linspace(0, 1, 100)
    axes[1].set_title('Posterior')
    axes[1].plot(gammas, list(map(f, gammas)))  
    plt.show()   















