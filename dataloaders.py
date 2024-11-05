### dataloaders.py ###


import numpy as np
import os
from pathlib import Path
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


# matched-pairs dataset class for scikit-learn
class SKLMatchedPairsCGL():
    """
    CGL matched-pairs dataloader for scikit-learn

    Parameters
    ----------
    data0_dir : str or PosixPath
        Name of directory containing steady-state samples
        Filenames should match those of data1_dir

    data1_dir : str or PosixPath
        Name of directory containing transient samples
        Filenames should match those of data0_dir

    normalize : bool, optional
        Whether or not to normalize samples
    
    split_fracs : np.ndarray, optional
        Proportions of data to allocate to training, development, and test sets

    seed : int, optional
        Randomization seed for splitting dataset
    
    Attributes
    ----------
    n_samples : int
        Number of samples

    n_features : int
        Number of features per sample  

    X_train, X_dev, X_test : np.ndarray
        Training, development, and test set features
    
    y_train, y_dev, y_test : np.ndarray
        Training, development, and test set labels

    """
    def __init__(self, data0_dir, data1_dir, normalize=True, split_fracs=(0.6, 0.2, 0.2), seed=1234):
        self.data0_dir = Path(data0_dir)
        self.data1_dir = Path(data1_dir)

        self.data_fns = np.sort([i for i in os.listdir(data0_dir) if '.mat' in i])
        # check that filenames are the same in two classes
        assert np.all(self.data_fns == np.sort([i for i in os.listdir(data1_dir) if '.mat' in i]))
        
        # load example sample to get number of features
        self.n_features = len(load_cgl_sample(self.data0_dir.joinpath(self.data_fns[0]),
                                            normalize=False, flatten=True))
        self.n_samples = len(self.data_fns)

        self.normalize = normalize
        self.split_fracs = split_fracs
        self.seed = seed
        
        self._load()
        self._split()
        
    def _load(self):
        '''Load samples'''
        # preallocate space
        data0 = np.zeros((self.n_samples, self.n_features), dtype='float32')
        data1 = np.zeros((self.n_samples, self.n_features), dtype='float32')

        for i in range(self.n_samples): # iterate over samples and fill data0 and data1
            data0[i] = load_cgl_sample(self.data0_dir.joinpath(self.data_fns[i]),
                                       normalize=self.normalize, flatten=True)
            data1[i] = load_cgl_sample(self.data1_dir.joinpath(self.data_fns[i]),
                                       normalize=self.normalize, flatten=True)
        
        self._data0 = data0
        self._data1 = data1

    def _split(self):
        '''Split into train, development, and test sets'''
        labels = get_split_idcs(self.n_samples, fractions=self.split_fracs, seed=self.seed)

        self.X_train = np.concatenate((self._data0[labels[0]], self._data1[labels[0]]))
        self.X_dev = np.concatenate((self._data0[labels[1]], self._data1[labels[1]]))
        self.X_test = np.concatenate((self._data0[labels[2]], self._data1[labels[2]]))

        del self._data0
        del self._data1

        self.y_train = np.concatenate((np.zeros(np.sum(labels[0])), np.ones(np.sum(labels[0])))).astype(int)
        self.y_dev = np.concatenate((np.zeros(np.sum(labels[1])), np.ones(np.sum(labels[1])))).astype(int)
        self.y_test = np.concatenate((np.zeros(np.sum(labels[2])), np.ones(np.sum(labels[2])))).astype(int)


### HELPERS ###


def load_cgl_sample(filepath, normalize=True, flatten=False):
    '''Load sample CGL frame from MATLAB matrix

    Parameters
    ----------
    filepath : str or PosixPath
        Path to CGL frame MATLAB matrix

    normalize : bool, optional
        Whether to scale amplitudes and phases to (0, 1]

    flatten : bool, optional
        Whether to flatten frame to vector of amplitudes and phases

    Returns
    -------
    sample : np.ndarray
        Loaded CGL sample
        If flatten is False, first slice is amplitudes and second is phases
        If flatten is True, first elements are amplitudes followed by phases
    '''
    raw_sample = loadmat(filepath)['data'] # load raw data from MATLAB matrix
    amps = np.abs(raw_sample) # load amplitudes
    phis = np.angle(raw_sample)+np.pi # load phases and scale to (0, 2pi]

    if normalize: # normalize amplitudes and phases to (0, 1]
        amps = amps-np.min(amps)
        amps = amps/np.max(amps)
        phis = phis/(2*np.pi)

    sample = np.stack((amps, phis)) # stack channels
    if flatten: sample = sample.flatten() # flatten to vector

    return sample


def get_split_idcs(n_samples, fractions=(0.6, 0.2, 0.2), seed=1234):
    '''Indices that dataset into training, development, and test sets

    Parameters
    ----------
    n_samples : int
        Number of samples

    fractions : np.ndarray, optional
        Array of floats that sum to 1

    seed : int, optional
        Randomization seed

    Returns
    -------
    split_idcs : np.ndarray
        Array of bools with one True per column
        Rows are splits
    '''
    # get sizes of datasets in samples
    set_sizes = (n_samples*np.array(fractions)).astype(int)
    set_sizes[0] += n_samples-np.sum(set_sizes) # round up largest

    # initialize destination dataset labels for samples
    set_labels = [i*np.ones(set_sizes[i]) for i in range(len(set_sizes))]
    set_labels = np.concatenate(set_labels)

    # shuffle destination labels to randomize samples to datasets
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(set_labels)

    split_idcs = np.array([set_labels == i for i in range(len(set_sizes))])
    return split_idcs
