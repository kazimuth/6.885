# Helper for merging traces
def merge(a, b, path=None):
    a = dict(a)
    if path is None: path = []
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            a[key] = merge(a[key], b[key], path + [str(key)])
        else:
            a[key] = b[key]
    return a
    
# Minimal GFI example
class GenerativeFunction():
    def __call__(self, *args):
        raise(NotImplementedError())
    def generate(self, args, observations):
        raise(NotImplementedError())

class PythonCodeGenerativeFunction(GenerativeFunction):
    def __init__(self, f):
        self.f = f
    
    def __call__(self, *args):
        def regular_sample(distribution, name):
            return distribution.sample()
        def regular_call(other_fn, name, args):
            return other_fn(*args)

        return inject_variables({'sample': regular_sample, 'call': regular_call}, self.f)(*args)
    
    def generate(self, args, observations={}):
        choices = {}
        score = 0

        def propose(distribution, name):
            nonlocal score
            if name in observations:
                choices[name] = observations[name]
                score += distribution.score(choices[name])
            else:
                choices[name] = distribution.sample()
            return choices[name]
        def call(other_fn, namespace, args=[]):
            nonlocal score
            subobs = observations[namespace] if namespace in observations else {}
            result = other_fn.generate(args, subobs)
            choices[namespace] = result['choices']
            score += result['score']
            return result['retval']
        
        retval = inject_variables({'sample': propose, 'call': call}, self.f)(*args)
        return {'retval': retval, 'choices': choices, 'score': score}

def compile_generative_function(f):
    return PythonCodeGenerativeFunction(f)

@compile_generative_function
def f1():
    p = sample(Uniform(), "p")
    flips = call(f2, "flips", args=[p, 10])
    return flips

@compile_generative_function
def f2(p, n):
    xs = []
    for i in range(n):
        xs.append(sample(Bernoulli(p), f'{i}'))
    return xs

print(f1())
print(f2(0.3, 10))
print(f1.generate([], {'p': 0.1, 'flips': {'1': True}}))

def importance_resampling_gfi(f, obs, n, args=[]):
    particles = [f.generate(args, obs) for _ in range(n)]
    return choose_particle(particles)

print(importance_resampling_gfi(f1, {'flips': {'0': True, '1': True, '2': True, '3': True}}, 100))