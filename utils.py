import numpy as np
from scipy.optimize import root_scalar, minimize
from scipy.signal import find_peaks

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






