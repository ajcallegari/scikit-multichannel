import ray
import os

default_redis_password = 'gn8GWVrMJ5cSX4'

def is_in(obj, iterable):
    """Determine if an obj instance is in a list. Needed to prevent Python from using 
       __eq__ method which throws errors for ndarrays
    """
    for element in iterable:
        if element is obj:
            return True
    return False

def get_index(obj, iterable):
    """Find the index of the first instance of an obj in a list. Needed to prevent Python from using 
       __eq__ method which throws errors for ndarrays
    """    
    index = 0
    for element in iterable:
        if element is obj:
            return index
        index += 1
    raise ValueError('object not found in the iterable')

class RayDistributor:
    """
    Class that provides ray distributed computing to pipecaster users.
    
    Examples
    -------

    (1) add remote computers to your resource pool

    import pipecaster as pc
    pc.connect_remote_computer(computer_id='ellen', app_path='/usr/venv/bin/', n_cpus='all', n_gpus=0, 
                               object_store_memory='auto')

    (2) use your own ray cluster for pipecaster parallel computing

    import pipecaster as pc
    pc.set_distributor(
    
    """
    
    def __init__(self):
        self._is_started = False
        
    def is_started(self):
        if self._is_started == False or ray.is_initialized() == False:
            return False
        else:
            return True   
        
    def startup(self, n_cpus='all', n_gpus='all', object_store_memory='auto', redis_password=default_redis_password):
        n_cpus = None if n_cpus == 'all' else n_cpus
        n_gpus = None if n_gpus == 'all' else n_gpus
        object_store_memory = None if object_store_memory == 'auto' else object_store_memory
        self.cpus = n_cpus
        self.gpus = n_gpus
        self.memory = object_store_memory
        self.redis_password = redis_password
        self.remote_nodes = []
        
        if ray.is_initialized():
            ray.shutdown()
            
        self.info = ray.init(_redis_password=self.redis_password, 
                             num_cpus=self.cpus, num_gpus=self.gpus,
                             object_store_memory=self.memory)
        
        self.redis_address = self.info['redis_address']
        self.ip_address = self.info['node_ip_address']
        self.dashboard_url = self.info['webui_url']
        
        if ray.is_initialized() == False:
            raise utils.ParallelBackendError('Ray initialization failed')
            
        self._is_started = True
        
    def get_info(self):
        return ray.nodes()
        
    def get_network_address(self):
        return self.redis_address
    
    def get_ip_address(self):
        return self.ip_address
    
    def get_dashboard_url(self):
        return self.dashboard_url
    
    def _start_remote_node(self, computer_id, app_path=None, n_cpus='all', n_gpus='all',
                            object_store_memory='auto'):
        command_string = 'ssh {} '.format(computer_id)
        if app_path is not None:
            command_string += app_path
        command_string += 'ray start --address={} --redis-password={} --redis-port={}'.format(self.redis_address,
                                                                                              self.redis_password,
                                                                                              self.redis_port)
        if n_cpus != 'all':
            command_string += ' --num-cpus={}'.format(n_cpus)
        if n_gpus != 'all':
            command_string += ' --num-gpus={}'.format(n_gpus)   
        if object_store_memory != 'auto':
            command_string += ' --object-store-memory={}'.format(object_store_memory)
        os.system(command_string)
        
    def connect_remote_node(self, computer_id, app_path=None, n_cpus='all', n_gpus='all', 
                          object_store_memory='auto'):
        """
        Add CPUs and GPUs from a from remote computer to pipecaster's parallel computing resources

        Notes
        -----
        The remote computer should typically be on the LAN and behind any firewalls.  
        To distribute jobs through a firewall is currently not possible because the ray backend 
        was not designed for this due to the lack of encryption for redis communications.

        Parameters
        ----------
        computer_id: string
            IP address of remote computer or ssh server id. 

        app_path: string, default=None
            Optional path to folder conataining the backend computing application on remote server.

        n_cpus: 'all' or int, default='all'
            Request this many CPUs from the remote computer.

        n_gpus: 'all' or float, default=0
            Request this many GPUs or fractional GPUs from the remote computer.

        object_store_memory: iterable, default='auto' 
            Number of bytes to reserve on the remote computer for the plasma in-memory object store.
        """        
        self._start_remote_node(computer_id, app_path, n_cpus, n_gpus, object_store_memory)
        self.remote_nodes.append((computer_id, app_path, n_cpus, n_gpus, object_store_memory))

    def shutdown(self):
        ray.shutdown()
        for computer_id, app_path, _, _, _ in self.remote_nodes:
            command_string =  'ssh {} '.format(computer_id)
            if app_path is not None:
                command_string += app_path
            command_string += 'ray stop'.format(computer_id, app_path)
            os.system(command_string)
        self._is_started = False

    def count_cpus(self):
        return int(sum([d['Resources']['CPU'] for d in self.get_info()]))

    def count_gpus(self):
        return sum([d['Resources']['GPU'] for d in self.get_info() if 'GPU' in d['Resources']])
   
    def store_shared_mem(self, obj):
        return ray.put(obj)
    
    def distribute_shared_objects(self, shared_mem_objects, *arg_lists):
        if shared_mem_objects is None or len(shared_mem_objects) == 0:
            return arg_lists
        else:
            plasma_objects = [ray.put(obj) for obj in shared_mem_objects]
            arg_lists = [[plasma_objects[get_index(obj, shared_mem_objects)] if is_in(obj, shared_mem_objects) 
                          else obj for obj in arg_list] for arg_list in arg_lists]
            return arg_lists
    
    def starmap_jobs(self, f, args_list, n_cpus='max', shared_mem_objects=None):
        '''Compute a list of jobs in parallel, where each job is specified by an args tuple in a single list

        Parameters
        ----------
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
        return self.map_jobs(f, *arg_lists, n_cpus=n_cpus, shared_mem_objects=shared_mem_objects)

    def map_jobs(self, f, *arg_lists, n_cpus='max', shared_mem_objects=None):
        
        '''Compute a list of jobs in parallel, where each job is specified by an set of lists, 
           with each list containing a different argument (standard Python map argument structure)
           
        Parameters
        ----------
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

        available_cpus = self.count_cpus()
        if available_cpus < 1:
            raise utils.ParallelBackendError('no cpus detected')
        n_cpus = available_cpus if (n_cpus == 'max' or n_cpus >= available_cpus) else n_cpus
        arg_lists = self.distribute_shared_objects(shared_mem_objects, *arg_lists)

        args_list = list(zip(*arg_lists))
        del arg_lists
        n_jobs = len(args_list)
        ray_f = ray.remote(f)
        
        if n_jobs >= n_cpus:
            # print('running {} processes, {} jobs using ray.get'.format(n_cpus, n_jobs))
            jobs = [ray_f.remote(*args) for args in args_list]
            results = ray.get(jobs)
        else:
            # throttle CPU utilization with chunks to conform to sklearn n_jobs interface
            chunk_size = n_cpus
            # print('running {} processes, {} jobs using chunked ray.get'.format(n_cpus, n_jobs))
            args_chunks = [args_list[i : i + chunk_size] for i in range(0, n_jobs, chunk_size)]
            results = []
            for args_chunk in args_chunks:
                job_chunk = [ray_f.remote(*args) for args in args_chunk]
                results.extend(ray.get(job_chunk))

        return results