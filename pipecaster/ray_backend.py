"""
Ray multiprocessing back end for :mod:`pipecaster.parallel`.
"""

import os
import ray

os.system('ulimit -n 8192')
default_redis_password = 'gn8GWVrMJ5cSX4'


def is_in(obj, iterable):
    """
    Determine if an obj instance is in a list. Needed to prevent Python from
    using __eq__ method which throws errors for ndarrays.
    """
    for element in iterable:
        if element is obj:
            return True
    return False


def get_index(obj, iterable):
    """
    Find the index of the first instance of an obj in a list. Needed to prevent
    Python from using __eq__ method which throws errors for ndarrays
    """
    index = 0
    for element in iterable:
        if element is obj:
            return index
        index += 1
    raise ValueError('object not found in the iterable')


class RayDistributor:
    """
    Ray distributed computing back end for :mod:`pipecaster.parallel`.

    Examples
    --------
    Building a cluster with an ssh computer id:
    ::

        import pipecaster as pc
        import pipecaster.ray_backend as ray_backend

        distributor = ray_backend.RayDistributor()
        distributor.connect_remote_computer(computer_id='ellen',
                                   app_path='/home/john/venv/bin/ray',
                                   n_cpus='all', n_gpus=0,
                                   object_store_memory='auto')
        pc.set_distributor(distributor)

    Notes
    -----
        - Python and ray versions must be identical on all cluster computers.
        - Ray makes it very hard to set communication ports, so it's not
          convenient to distribute jobs through a firewall.  Typically cluster
          computers will have no individual firewalls but will be protected
          from external attacks by a router firewall.
    """

    def __init__(self):
        self._is_started = False

    def is_started(self):
        """
        Detect if the distributor has been started.
        """
        if self._is_started is False or ray.is_initialized() is False:
            return False
        else:
            return True

    def startup(self, n_cpus='all', n_gpus=None, object_store_memory='auto',
                redis_password=default_redis_password):
        """
        Start the local ray head node.  If ray has already been started, shut
        it down and restart.

        Parameters
        ----------
        n_cpus: int or 'all', default='all'
            - The number of CPUs to request from the local computer.
            - If int: request n_cpus number of cpus
            - If 'all': request all CPUs
        n_gpus: None, int or 'all', default=None
            - The number of GPUs to request from the local computer.
            - If None: no GPUs will be requested
            - If int: n_gpus number of GPUs will be requested
            - If 'all': request all availalbe GPUs
        object_store_memory: int or 'auto'
            The number of bytes to request from the in-memory shared object
            store.
        redis_password: str, default=default_redis_password
            Set the password used to protect redis ports from attacks.
        """
        n_cpus = None if n_cpus == 'all' else n_cpus
        n_gpus = 0 if n_gpus is None else n_gpus
        n_gpus = None if n_gpus == 'all' else n_gpus
        object_store_memory = (None if object_store_memory == 'auto'
                               else object_store_memory)
        self.cpus = n_cpus
        self.gpus = n_gpus
        self.memory = object_store_memory
        self.redis_password = redis_password
        self.remote_computers = []

        if ray.is_initialized():
            ray.shutdown()

        self.info = ray.init(_redis_password=self.redis_password,
                             num_cpus=self.cpus, num_gpus=self.gpus,
                             object_store_memory=self.memory)

        self.redis_address = self.info['redis_address']
        self.ip_address = self.info['node_ip_address']
        self.dashboard_url = self.info['webui_url']

        if ray.is_initialized() is False:
            raise utils.ParallelBackendError('Ray initialization failed')

        self._is_started = True

    def start_if_needed(self, n_cpus='all', n_gpus='all',
                        object_store_memory='auto',
                        redis_password=default_redis_password):
        if self.is_started() is False:
            self.startup(n_cpus='all', n_gpus='all',
                         object_store_memory='auto',
                         redis_password=default_redis_password)

    def get_info(self):
        """
        Get a dict with information about the cluster's addresses and
        resources.
        """
        return ray.nodes()

    def get_network_address(self):
        """
        Get the network address used for communcation between networked
        computers.
        """
        return self.redis_address

    def get_ip_address(self):
        return self.ip_address

    def get_dashboard_url(self):
        """
        Get the URL for Ray's resource usage dashboard.
        """
        return self.dashboard_url

    def _start_remote_computer(self, computer_id, app_path, n_cpus='all',
                               n_gpus='all', object_store_memory='auto'):
        command_string = 'ssh {} '.format(computer_id)
        command_string += app_path
        command_string += (' start --address={} --redis-password={}'
                           .format(self.redis_address, self.redis_password))

        if n_cpus != 'all':
            command_string += ' --num-cpus={}'.format(n_cpus)
        if n_gpus != 'all':
            command_string += ' --num-gpus={}'.format(n_gpus)
        if object_store_memory != 'auto':
            command_string += ' --object-store-memory={}'.format(
                                                        object_store_memory)
        os.system(command_string)

    def connect_remote_computer(self, computer_id, app_path=None, n_cpus='all',
                                n_gpus=None, object_store_memory='auto'):
        """
        Start ray on a remote computer and add requested resources to the
        cluster.

        Parameters
        ----------
        computer_id: string
            ssh server id.
        app_path: string
            Path to the remote ray executable.
        n_cpus: int or 'all', default='all'
            - The number of CPUs to request from the local computer.
            - If int: request n_cpus number of cpus
            - If 'all': request all CPUs
        n_gpus: None, int or 'all', default=None
            - The number of GPUs to request from the local computer.
            - If int: n_gpus number of GPUs will be requested
            - If 'all': request all availalbe GPUs
        object_store_memory: int or 'auto'
            The number of bytes to request from the in-memory shared object
            store.
        redis_password: str, default=default_redis_password
            Set the password used to protect redis ports from attacks.
        """
        self.start_if_needed()
        self._start_remote_computer(computer_id, app_path, n_cpus, n_gpus,
                                    object_store_memory)
        self.remote_computers.append((computer_id, app_path, n_cpus, n_gpus,
                                      object_store_memory))

    def shutdown(self):
        """
        Stop all ray instances running locally or remotely and free all
        resources.
        """
        ray.shutdown()
        for computer_id, app_path, _, _, _ in self.remote_computers:
            command_string = 'ssh {} '.format(computer_id)
            command_string += app_path
            command_string += ' stop'.format(computer_id, app_path)
            os.system(command_string)
        self._is_started = False

    def count_cpus(self):
        """
        Count all CPUs available to the cluster.
        """
        return int(sum([d['Resources']['CPU'] for d in self.get_info()]))

    def count_gpus(self):
        """
        Count all GPUs available to the cluster.
        """
        return sum([d['Resources']['GPU'] for d in self.get_info()
                    if 'GPU' in d['Resources']])

    def store_object(self, obj):
        """
        Manually store an object in the in-memory shared object store.
        Returns a reference to the remote object.
        """
        return ray.put(obj)

    def distribute_shared_objects(self, shared_mem_objects, *arg_lists):
        """
        Pre-store designated objects in shared memory and replace references to
        those objects with references to the shared objects every time they
        appear in a list of arguments.  This step prevents ray from repeatedly
        sending those objects to the store with each function call.

        Parameters
        ----------
        shared_mem_objects : list
            The objects designated for shared memory storage.
        arg_lists : variable number of lists
            Each list contains a list of objects.
        """
        if shared_mem_objects is None or len(shared_mem_objects) == 0:
            return arg_lists
        else:
            plasma_objects = [ray.put(ob)
                              if (type(ob) != ray._raylet.ObjectRef)
                              else ob
                              for ob in shared_mem_objects]
            arg_lists = [[plasma_objects[get_index(obj, shared_mem_objects)]
                          if is_in(obj, shared_mem_objects)
                          else obj for obj in arg_list]
                         for arg_list in arg_lists]
            return arg_lists

    def starmap_jobs(self, f, args_list, n_cpus='max',
                     shared_mem_objects=None):
        '''
        Compute a list of jobs in parallel. Call signature similar to the
        Python standard library multiprocessing starmap() function.

        Arguments
        ---------
        f : callable
            Python callable object that you want to execute in parallel.
        arg_lists : list of tuples
            List of argument tuples or lists, where each tuple/list specifies a
                job.
            e.g. [(arg1-job1, arg2-job1, arg3-job1),
                  (arg1-job2, arg2-job2, arg3-job2)]
        n_cpus : 'max' or int, default='max'
            Number of parallel processes to use for the jobs.
            'max' requests all available CPUs.
        shared_mem_objects : list or None, default=None
            List of Python objects to pre-store in the plasma in-memory object
            store to prevent repeated storage of large objects.

        Examples
        --------
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

            output:
            [2.0, 3.0, 4.0]
        '''
        arg_lists = list(zip(*args_list))
        return self.map_jobs(f, *arg_lists, n_cpus=n_cpus,
                             shared_mem_objects=shared_mem_objects)

    def map_jobs(self, f, *arg_lists, n_cpus='max', shared_mem_objects=None):
        '''
        Compute a list of jobs in parallel using a signature like that of the
        standard Python map() function.

        Parameters
        ----------
        f : callable
            Python callable object that you want to execute in parallel.
        *arg_lists : iterable
            List of arg_lists, where each arg_list is list of arguments to be
            sent to f() ordered by job number.
            e.g. [arg1-job1, arg1-job2, arg1-job3],
                 [arg2-job1, arg2-job2, arg2-job3]
        n_cpus : 'max' or int, default='max'
            Number of parallel processes to use for the jobs.  'max' requests
            all available CPUs.
        shared_mem_objects : iterable or None, default=None
            List of Python objects to pre-store in the plasma in-memory object
            store to prevent repeated storage of large objects.

        Examples
        --------
        ::

            import pipecaster.parallel as parallel

            def f(a, big_arg):
                return a + big_arg

            As = [1, 2, 3]
            Bs = [big_arg, big_arg, big_arg]

            args_list = zip(As, Bs)
            results = parallel.starmap_jobs(f, args_list, n_cpus='max',
                                            shared_mem_objects=[big_arg])

        note: Using shared_mem_objects only increases speed when a large
        argument is passed multiple times in the jobs list because in those
        instances ray will copy the argument to the plasma store each time the
        argument is passed, whereas shared_mem_objects guarantees that the
        object will only be transferred to the object store one time.
        '''
        available_cpus = self.count_cpus()
        if available_cpus < 1:
            raise utils.ParallelBackendError('no cpus detected')
        n_cpus = (available_cpus if
                  (n_cpus == 'max' or n_cpus >= available_cpus) else n_cpus)
        arg_lists = self.distribute_shared_objects(shared_mem_objects,
                                                   *arg_lists)

        args_list = list(zip(*arg_lists))
        del arg_lists
        n_jobs = len(args_list)
        ray_f = ray.remote(f)

        if n_jobs >= n_cpus:
            jobs = [ray_f.remote(*args) for args in args_list]
            results = ray.get(jobs)
        else:
            chunk_size = n_cpus
            args_chunks = [args_list[i: i + chunk_size]
                           for i in range(0, n_jobs, chunk_size)]
            results = []
            for args_chunk in args_chunks:
                job_chunk = [ray_f.remote(*args) for args in args_chunk]
                results.extend(ray.get(job_chunk))

        return results
