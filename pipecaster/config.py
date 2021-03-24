"""
Configuration options for pipecaser.
"""

# Set of methods recognized by pipecaster as prediction methods.
recognized_pred_methods = set(['predict', 'predict_proba',
                               'decision_function', 'predict_log_proba'])

# Set the order in which methods are automatically selected for
# inferring.
predict_method_precedence = ['predict', 'predict_proba',
                             'predict_log_proba', 'decision_function']

# Set the order in which methods are automatically selected for
# performance scoring.
score_method_precedence = ['predict_proba', 'predict_log_proba',
                           'decision_function', 'predict']

# Set the order in which methods are automatically selected for
# transforming with a predictor.
transform_method_precedence = ['predict_proba', 'predict_log_proba',
                               'decision_function', 'predict']

# Set the distributed computing backend for parallel.py module.
from pipecaster.ray_backend import RayDistributor
default_distributor_type = RayDistributor
