import random
from sklearn.datasets import make_classification

def make_multi_input_classification(n_classes = 2, 
                                    n_Xs=10, n_informative_Xs=3, n_weak_Xs=0,
                                    n_samples=1000, 
                                    n_features=500, 
                                    n_informative=100,
                                    n_redundant=0,
                                    n_repeated=0,
                                    class_sep=1.0,
                                    weak_noise_sd=50):
    
    assert (n_informative_Xs + n_weak_Xs) <= n_Xs, 'too many informative & weak marices for n_Xs'
        
    n_random_Xs = n_Xs - n_informative_Xs - n_weak_Xs
    
    X, y = make_classification(n_samples=n_samples, 
                               n_features=n_features * (n_informative_Xs + n_weak_Xs), 
                               n_informative=n_informative * (n_informative_Xs + n_weak_Xs), 
                               n_redundant=n_redundant * (n_informative_Xs + n_weak_Xs),
                               n_repeated=n_repeated * (n_informative_Xs + n_weak_Xs),
                               class_sep=class_sep)
    
    rand_X = np.random.normal(loc=50, scale=20, 
                              size=(n_samples, n_features * n_random_Xs))
    
    # split synthetic data into separate matrices
    Xs = [X[:, i*n_features:(i+1)*n_features] for i in range(n_informative_Xs + n_weak_Xs)]
    rand_Xs = [rand_X[:, i*n_features:(i+1)*n_features] for i in range(n_random_Xs)]
    
    # add extra gaussian noise to create weak matrices
    for i in range(n_weak_Xs):
        Xs[i] += np.random.normal(loc=50, scale=weak_noise_sd, size=(n_samples,n_features))
    X_types = ['weak' for i in range(n_weak_Xs)] 
    X_types += ['informative' for i in range(n_informative_Xs)]
    X_types += ['random' for i in range(n_random_Xs)] 
    Xs = Xs + rand_Xs
    tuples = list(zip(Xs, X_types))
    random.shuffle(tuples)
    Xs, X_types = zip(*tuples)
    
    return list(Xs), y, list(X_types)