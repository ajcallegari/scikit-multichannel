"""
Distributed computing interface.

Users can typically ignore this module unless they want to get information
about resources or set a custom back end using
set_distributor(my_backend_instance).  For info on building a cluster, see
:class:`pipecaster.ray_backend.RayDistributor`.

Notes
-----
Custom parallel backend implementations need to provide a distributor Class
with these methods:
    - map_jobs(f, *arg_lists, n_cpus='max', shared_mem_objects=None)
    - is_started() : takes no args, indicates if start() needs to be called
      before jobs can be executed
    - startup(n_cpus='all', n_gpus='all', object_store_memory='auto')
    - count_cpus() : return total number of CPUs available for computing
    - count_gpus() : return total number of GPUs available for computing
    - shutdown() : free up the computing resources of the Distributor

Examples
--------
basic multiprocessing with map interface:
::

    import pipecaster.parallel as parallel

    def f(a, b):
        return a + b

    As = [1, 2, 3]
    Bs = [4, 5, 6]

    results = parallel.map_jobs(f, As, Bs, n_cpus='max')
    print(results)

    output:
    [5, 7, 9]

basic multiprocessing with starmap interface:
::

    import pipecaster.parallel as parallel

    # same as example 2 but add zip() and change the map call as follows:
    args_list = zip(As, Bs)
    results = parallel.starmap_jobs(f, args_list, n_cpus='max')
    print(results)

    output:
    [5, 7, 9]

fast distribution of large objects for multiprocessing using a shared memory
stores:
::

    import pipecaster.parallel as parallel
    import numpy as np

    def f(a, big_arg):
        return a + np.mean(big_arg)

    big_arg = np.ones(1000)

    As = [1, 2, 3]
    Bs = [big_arg, big_arg, big_arg]

    args_list = zip(As, Bs)
    results = parallel.starmap_jobs(f, args_list, n_cpus='max',
                                    shared_mem_objects=[big_arg])
    print(results)

    # output: [2.0, 3.0, 4.0]
"""

import os
import ray
import multiprocessing
import time

import pipecaster.config as config
import pipecaster.utils as utils


__all__ = ['set_distributor', 'starmap_jobs', 'map_jobs', 'count_local_cpus',
           'count_cpus', 'count_gpus']

default_distributor_type = config.default_distributor_type
distributor_type = default_distributor_type
distributor = None


def set_distributor(distributor_):
    """
    Set a custom distributor.  Should be an instantiated RayDistributor
        instance or object with similar interface.
    """
    global distributor
    distributor = distributor_


def startup(n_cpus='all', n_gpus=None, object_store_memory='auto'):
    """
    Start a multiprocessing back end.

    Parameters
    ----------
    n_cpus: 'all' or int, default='all'
        Number of CPUs to request at startup.
    n_gpus: int or 'all', default=None
        Number of GPUs to request at startup
    object_store_memory: int or 'auto', default='auto'
        Number of bytes of memory to request for the shared object store.
    """
    global distributor
    if distributor is not None:
        distributor.startup(n_cpus, n_gpus, object_store_memory)
    else:
        distributor = default_distributor_type()
        distributor.startup(n_cpus, n_gpus, object_store_memory)


def start_if_needed(n_cpus='all', n_gpus='all', object_store_memory='auto'):
    """
    Start a multiprocessing back end if one has not already been started.

    Parameters
    ----------
    n_cpus: 'all' or int, default='all'
        Number of CPUs to request at startup.
    n_gpus: int or 'all', default=None
        Number of GPUs to request at startup
    object_store_memory: int or 'auto', default='auto'
        Number of bytes of memory to request for the shared object store.
    """
    if distributor is None or distributor.is_started() is False:
        startup(n_cpus, n_gpus, object_store_memory)


def shutdown():
    """
    Shut the parallel back end down.
    """
    if distributor is not None and distributor.is_started():
        distributor.shutdown()


def count_local_cpus():
    """
    Get the number of CPUs on the local computer (where __main__ is located).
    """
    return multiprocessing.cpu_count()


def count_cpus():
    """
    Get the total number of CPUs available in the parallel back end including
    those of both the local computer and remote computers in the cluster.
    """
    if distributor is None or distributor.is_started() is False:
        return count_local_cpus()
    else:
        return distributor.count_cpus()


def count_gpus():
    """
    Get the total number of GPUs available in the parallel back end including
    those of both the local computer and remote computers in the cluster.
    """
    if distributor is None or distributor.is_started() is False:
        return count_local_cpus()
    else:
        return distributor.count_gpus()


def starmap_jobs(f, args_list, n_cpus='max', shared_mem_objects=None):
    '''
    Compute a list of jobs in parallel. Call signature similar to the
    Python standard library multiprocessing starmap() function.

    Arguments
    ---------
    f: callable
        Python callable object that you want to execute in parallel.
    arg_lists: list of tuples
        List of argument tuples or lists, where each tuple/list specifies a
            job.
        e.g. [(arg1-job1, arg2-job1, arg3-job1),
              (arg1-job2, arg2-job2, arg3-job2)]
    n_cpus: 'max' or int, default='max'
        Number of parallel processes to use for the jobs.
        'max' requests all available CPUs.
    shared_mem_objects: iterable or None, default=None
        List of Python objects to pre-store in the plasma in-memory object
        store to prevent repeated storage of large objects.

    Example
    -------
    import pipecaster.parallel as parallel
    import numpy as np

    def f(a, big_arg):
        return a + np.mean(big_arg)

    big_arg = np.ones(1000)

    As = [1, 2, 3]
    Bs = [big_arg, big_arg, big_arg]

    args_list = zip(As, Bs)
    results = parallel.starmap_jobs(f, args_list, n_cpus='max',
                                    shared_mem_objects=[big_arg])
    print(results)

    output:
    [2.0, 3.0, 4.0]
    '''
    arg_lists = list(zip(*args_list))
    return map_jobs(f, *arg_lists, n_cpus=n_cpus,
                    shared_mem_objects=shared_mem_objects)


def map_jobs(f, *arg_lists, n_cpus='max', shared_mem_objects=None):
    '''
    Compute a list of jobs in parallel using a signature like that of the
    standard Python map() function.

    Parameters
    ----------
    f: callable
        Python callable object that you want to execute in parallel.
    *arg_lists: iterable
        List of arg_lists, where each arg_list is list of arguments to be sent
        to f() ordered by job number.
        e.g. [arg1-job1, arg1-job2, arg1-job3],
             [arg2-job1, arg2-job2, arg2-job3]
    n_cpus: 'max' or int, default='max'
        Number of parallel processes to use for the jobs.  'max' requests all
        available CPUs.
    shared_mem_objects: iterable or None, default=None
        List of Python objects to pre-store in the plasma in-memory object
        store to prevent repeated storage of large objects.

    Example
    -------
    import pipecaster.parallel as parallel

    def f(a, big_arg):
        return a + big_arg

    As = [1, 2, 3]
    Bs = [big_arg, big_arg, big_arg]

    args_list = zip(As, Bs)
    results = parallel.starmap_jobs(f, args_list, n_cpus='max',
                                    shared_mem_objects=[big_arg])

    note: Using shared_mem_objects only increases speed when a large argument
    is passed multiple times in the jobs list because in those instances ray
    will copy the argument to the plasma store each time the argument is
    passed, whereas shared_mem_objects guarantees that the object will only be
    transferred to the object store one time.
    '''
    start_if_needed()
    return distributor.map_jobs(f, *arg_lists, n_cpus=n_cpus,
                                shared_mem_objects=shared_mem_objects)
