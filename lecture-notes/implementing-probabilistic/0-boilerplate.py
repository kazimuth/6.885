import random
import math
from scipy.special import logsumexp
from scipy.stats import norm
import numpy as np

# Helper: inject variables into a function's scope:
from functools import wraps
def inject_variables(context, func):
    @wraps(func)
    def new_func(*args, **kwargs):
        func_globals = func.__globals__
        saved_values = func_globals.copy()
        func_globals.update(context)
        try:
            result = func(*args, **kwargs)
        finally:
            for (var, val) in context.items():
                if var in saved_values:
                    func_globals.update({var: saved_values[var]})
                else:
                    del func_globals[var]
        return result
    return new_func

# Primitive Distributions
class Distribution: pass

class Bernoulli(Distribution):
    def __init__(self, p):
        self.p = p
    def sample(self):
        return random.random() < self.p
    def score(self, x):
        if x:
            return math.log(self.p)
        else:
            return math.log(1-self.p)

class Uniform(Distribution):
    def sample(self):
        return random.random()
    def score(self, x):
        return 0

class Normal(Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    def sample(self):
        return np.random.normal(self.mu, self.sigma)
    def score(self, x):
        return norm.logpdf(x, self.mu, self.sigma)
