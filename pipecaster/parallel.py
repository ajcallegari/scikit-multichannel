import os
import ray
import multiprocessing
import time

import pipecaster.utils as utils
from pipecaster.ray_backend import RayDistributor

"""Parallel computing backend for pipecaster.  Users can typically ignore this module 
   unless they want to get information about resources or set a custom backend 
   using set_distributor(my_backend_instance).  For info on building a cluster,
   the the ray_backend module.
   
Notes
-----
Custom parallel backend implementations need to provide a distributor Class with these methods:
1) map_jobs(f, *arg_lists, n_cpus='max', shared_mem_objects=None)
2) is_started(): takes no args, indicates if started need to be called before jobs can be taken
3) startup(n_cpus='all', n_gpus='all', object_store_memory='auto'): met
4) count_cpus(): return total number of CPUs available for computing
5) count_gpus(): return total number of GPUs available for computing
6) shutdown(): free up the computing resources of the Distributor

Examples
-------

(1) basic multiprocessing with map interface

import pipecaster.parallel as parallel

def f(a, b):
    return a + b
    
As = [1, 2, 3]
Bs = [4, 5, 6]
    
results = parallel.map_jobs(f, As, Bs, n_cpus='max')
print(results)

output:
[5, 7, 9]

(2) basic multiprocessing with starmap interface

# same as example 2 but add zip() and change the map call as follows:
args_list = zip(As, Bs)
results = parallel.starmap_jobs(f, args_list, n_cpus='max')
print(results)

output:
[5, 7, 9]

(3) fast distribution of large objects for multiprocessing using a shared memory stores

import pipecaster.parallel as parallel

def f(a, big_arg):
    return a + big_arg
    
As = [1, 2, 3]
Bs = [big_arg, big_arg, big_arg]

args_list = zip(As, Bs)
results = parallel.starmap_jobs(f, args_list, n_cpus='max', shared_mem_objects=[big_arg])
print(results)

output:
[5, 7, 9]

"""

__all__ = ['set_distributor', 'starmap_jobs', 'map_jobs', 'count_local_cpus', 
           'count_cpus', 'count_gpus']

default_distributor_type = RayDistributor
distributor_type = default_distributor_type
distributor = None
        
def set_distributor(distributor_):
    """Set a custom distributor.  Should be an instantiated RayDistributor instance or object with similar interface. 
    """
    global distributor
    distributor = distributor_
    
def startup(n_cpus='all', n_gpus='all', object_store_memory='auto'):
    global distributor
    if distributor is not None:
        distributor.startup(n_cpus, n_gpus, object_store_memory)
    else:
        distributor = default_distributor_type()
        distributor.startup(n_cpus, n_gpus, object_store_memory)
            
def start_if_needed(n_cpus='all', n_gpus='all', object_store_memory='auto'):
    if distributor is None or distributor.is_started() == False:
        startup(n_cpus, n_gpus, object_store_memory)
        
def shutdown():
    if distributor is not None and distributor.is_started():
        distributor.shutdown()
        
def count_local_cpus():
    return multiprocessing.cpu_count()

def count_cpus():
    if distributor is None or distributor.is_started() == False:
        return count_local_cpus()
    else:
        return distributor.count_cpus()

def count_gpus():
    if distributor is None or distributor.is_started() == False:
        return count_local_cpus()
    else:
        return distributor.count_gpus()
    
def starmap_jobs(f, args_list, n_cpus='max', shared_mem_objects=None):
    '''
    Compute a list of jobs in parallel, where each job is specified by an args tuple in a single list
       
    Arguments
    ---------
    f: callable
        Python callable object that you want to execute in parallel.
        
    arg_lists: iterable
        List of argument tuples or lists, where each tuple/list specifies a job.
        e.g. [(arg1-job1, arg2-job1, arg3-job1), (arg1-job2, arg2-job2, arg3-job2)]
        
    n_cpus: 'max' or int, default='max'
        Number of parallel processes to use for the jobs.  'max' requests all available CPUs.
        
    shared_mem_objects: iterable or None, default=None
        List of Python objects to pre-store in the plasma in-memory object store to prevent 
        repeated storage of large objects.
    
    Example
    -------
    import pipecaster.parallel as parallel

    def f(a, big_arg):
        return a + b

    As = [1, 2, 3]
    Bs = [big_arg, big_arg, big_arg]

    args_list = zip(As, Bs)
    results = parallel.starmap_jobs(f, args_list, n_cpus='max', shared_mem_objects=[big_arg])
    print(results)

    output:
    [5, 7, 9]
    
    '''
    arg_lists = list(zip(*args_list))
    return map_jobs(f, *arg_lists, n_cpus=n_cpus, shared_mem_objects=shared_mem_objects)

def map_jobs(f, *arg_lists, n_cpus='max', shared_mem_objects=None):
    '''
    Compute a list of jobs in parallel, where each job is specified by an set of lists, with each list
    containing a different argument (standard Python map argument structure)
    
    Arguments
    ---------
    f: callable
        Python callable object that you want to execute in parallel.
        
    *arg_lists: iterable
        List of arg_lists, where each arg_list is list of arguments to be sent to f() ordered by job number.
        e.g. [[arg1-job1, arg1-job2, arg1-job3], [arg2-job1, arg2-job2, arg2-job3]]
        
    n_cpus: 'max' or int, default='max'
        Number of parallel processes to use for the jobs.  'max' requests all available CPUs.
        
    shared_mem_objects: iterable or None, default=None
        List of Python objects to pre-store in the plasma in-memory object store to prevent 
        repeated storage of large objects.
    
    Example
    -------
    import pipecaster.parallel as parallel

    def f(a, big_arg):
        return a + big_arg

    As = [1, 2, 3]
    Bs = [big_arg, big_arg, big_arg]

    args_list = zip(As, Bs)
    results = parallel.starmap_jobs(f, args_list, n_cpus='max', shared_mem_objects=[big_arg])

    note: This feature only increases speed when a large argument is passed multiple times in your jobs
        because in those instances ray will copy the argument to the plasma store each time the argument is passed, 
        whereas shared_mem_objects guarantees that the object will only be placed in the store one time.
    '''
    start_if_needed()       
    return distributor.map_jobs(f, *arg_lists, n_cpus=n_cpus, shared_mem_objects=shared_mem_objects)
