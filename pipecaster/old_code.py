class Layer_old2:
    """A list of pipe instances with input mappings that supports construction via slice assignment and broadcasting.
    
    Examples
    --------
    
    """
    
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.pipe_list = []
        self.all_inputs = np.arange(self.n_inputs)
        self.mapped_inputs = set()
        
    def _get_slice_indices(slice_):
        return self.all_inputs[slice_]
        
    def __setitem__(self, slice_, val):
        
        is_index = type(slice_) == int
        is_listlike = isinstance(val, (list, tuple, np.ndarray))
        
        if is_index == True:
            inputs = [is_index]
        else:
            if slice_.step not in [None, 1]:
                raise ValueError('Invalid slice step; must be exactly 1 (Pipes may only act on contiguous inputs)')
            inputs = _get_slice_indices(slice_)
        if len(self.mapped_inputs.intersection(set(inputs))) > 0:
            raise ValueError('Two pipes use the same input. This is prevented to keep a 1:1 map between inputs and oututs') 
        
        if is_index == True and is_listlike == False:
            self.pipe_list.append((get_clone(val), inputs))
        
        elif is_index and is_listlike:
            if len(val) == 1:
                self.pipe_list.append((get_clone(val[0]), inputs))
            else: 
                raise TypeError('Number of pipes assigned does not match slice dimension of 1')
                
        elif is_index == False and is_listlike == True:
            n = len(inputs)
            if n != len(inputs):
                raise TypeError('The number of pipes assigned does not match the indicated slice length of {}'.format(n))
            else:
                for pipe, i in zip(val, inputs):
                    self.pipe_list.append((get_clone(pipe), [i]))
                                
        elif is_index == False and is_listlike == False:
            n = len(inputs)
            if is_multi_input(val) == True:
                self.pipe_list.append((get_clone(val), inputs))
            else:
                for i in inputs:
                    self.pipe_list.append((get_clone(val), [i]))
            
        self.mapped_inputs.add(inputs)
        
        return self
        
    def __getitem__(self, key):
        pass
        
    def __getslice__(self, i, j):
        pass
        
        
        
class Layer_old:
    """An ordered array pipe instances that supports assignment and broadcasting by cloning.
    
    Examples
    --------
    

    """
    
    def __init__(self, n_inputs):
        self.pipes = np.array(['passthrough' for i in range(n_inputs)], dtype=object)
        
    def __setitem__(self, slice_, val):
        
        is_index = type(slice_) == int
        is_listlike = isinstance(val, (list, tuple, np.ndarray))
        
        
        if is_index == True and is_listlike == False:
            self.pipes[slice_] = get_clone(val)
        
        elif is_index and is_listlike:
            if len(val) == 1:
                self.pipes[slice_] = get_clone(val[0])
            else: 
                raise TypeError('Number of pipes assigned does not match slice dimension of 1')
                
        elif is_index == False and is_listlike == True:
            n = len(self.pipes[slice_])
            if n != len(val):
                raise TypeError('Number of pipes assigned does not match slice len of {}'.format(n))
            else:
                self.pipes[slice_] = [get_clone(v) for v in val]
                
        elif is_index == False and is_listlike == False:
            n = len(self.pipes[slice_])
            self.pipes[slice_] = [get_clone(val) for i in range(n)]
        
    def __getitem__(self, key):
        pass
        
    def __getslice__(self, i, j):
        pass