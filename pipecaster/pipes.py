

class PassThrough:
    
    def fit(self, X, y=None, **fit_params):
        pass
    
    def transform(self,X, y=None, **fit_params):
        return X, y
    
    def fit_transform(self,X, y=None, **fit_params):
        return X, y
    
    def get_params(self):
        return {}
    
    def set_params(self, **params):
        pass
    
class Pipe:
    
    def __init__(self, f):
        self.f = f
    
    def fit(X, y):
        self.f.fit(self, X, y=None, **fit_params)
    
    def transform(X, y):
        return self.f.transform(self, X, y=None, **fit_params)
    
    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(self, X, y=None, **fit_params)
    
    def get_params(self):
        return self.f.get_params()
    
    def set_params(self, **params):
        self.f.set_params(params)