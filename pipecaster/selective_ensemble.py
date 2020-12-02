import ray

import pipecaster.utils as utils


                                 
class SelectiveEnsemble:
    
    def __init__(self, predictors, model_scorer, score_selector, internal_cv=5, predictor_jobs=1, cv_jobs=1):
        self.predictors = predictors
        self.model_scorer = model_scorer
        self.score_selector = score_selector
        self.internal_cv = internal_cv
        self.predictor_jobs = predictor_jobs
        self.cv_jobs = cv_jobs
    
    @staticmethod
    def _get_score(model_scorer, predictor, X, y, **fit_params):
        if X is None:
            return None
        else:
            return model_scorer(model_scorer, X, y, **fit_params)
    
    @ray.remote
    def _ray_get_score(model_scorer, predictor, X, y, **fit_params):
        return _get_score(model_scorer, predictor, X, y, **fit_params)
    
    def fit(self, X, y=None, **fit_params):
        self.predictors = [utils.get_clone(p) for p in self.predictors]
        self.predictors = [TransformingPredictor(p) if X is not None else None for X, p in zip(Xs, self.predictors)]
            
        if self.channel_jobs < 2:
            performance_scores = [ChannelModelSelector._get_score(self.model_scorer, predictor, X, y, **fit_params) 
                                  for predictor, X in zip(self.predictors, Xs)]
        else:
            y = ray.put(y)
            model_scorer = ray.put(self.model_scorer)
            jobs = [ChannelModelSelector._ray_get_score.remote(model_scorer, ray.put(predictor), 
                                                               ray.put(X), y, **fit_params) 
                    for predictor, X in zip(self.predictors, Xs)]
            performance_scores = ray.get(jobs)
        
        selected_indices = set(self.score_selector(performance_scores))
        self.predictors = [self.predictors[i] if i in selected_indices else None for i in range(len(Xs))]
            
    def transform(self, Xs):
        return [p.transform(Xs) if p is not None else None for p in self.predictors]
            
    def fit_transform(self, Xs, y=None, **fit_params):
        self.fit(Xs, y, **fit_params)
        return [p.transform(Xs) if p is not None else None for p in self.predictors]
    
    def get_params(self, deep=False):
        return {'predictors':self.predictors,
                'model_scorer':self.model_scorer,
                'score_selector':self.score_selector,
                'channel_jobs':self.channel_jobs,
                'cv_jobs':self.cv_jobs}
    
    def set_params(self, **params):
        for key, value in params.items():
            if key in ['predictors', 'model_scorer','score_selector', 'channel_jobs', 'cv_jobs']:
                setattr(self, key, value)
            else:
                raise AttributeError('{} not a valid ChannelModelSelector parameter'.format(key))
                
    def get_selection_indices(self):
        return [i for i, p in enumerate(self.predictors) if p is not None]
                
    def get_clone(self):
        clone = ChannelModelSelector(utils.get_list_clone(self.predictors), self.model_scorer, 
                                     self.score_selector, self.channel_jobs, self.cv_jobs)
        return clone