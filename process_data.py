import numpy as np

def load_data(file):
    in_data = np.loadtxt(file, skiprows=1)

    data = np.column_stack((np.log(in_data[:,0]),
                            np.log(in_data[:,3]),
                            (in_data[:,1]+in_data[:,2])/in_data[:,0],
                            (in_data[:,4]+in_data[:,5])/in_data[:,3]))

    return data.view(np.dtype([('mobs', np.float),
                               ('proxy', np.float),
                               ('dmobs', np.float),
                               ('dproxy', np.float)]))
                            
