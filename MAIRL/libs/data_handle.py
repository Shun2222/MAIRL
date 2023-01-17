import pickle
import numpy as np

def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

def mean_pre_nex(log_file, key, num_pre=5, num_nex=5):
    log = pickle_load(log_file)
    data = log[key]
    m_data = np.mean(data, axis=0)[1:]
    ave_data = []
    for i in np.arange(len(m_data)):
        pre = i-num_pre if i-num_pre>=0 else 0
        nex = i+num_nex if i+num_nex<len(m_data) else len(m_data)-1
        ave_data += [np.mean(m_data[pre:nex])]
    return ave_data  