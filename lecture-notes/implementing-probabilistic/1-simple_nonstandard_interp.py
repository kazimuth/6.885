def f():
    p = sample(Uniform())
    xs = []
    for i in range(3):
        xs.append(sample(Bernoulli(p)))
    return sum(xs)

def run(f, args=[]):
    # Standard interpretation: when you see a `sample`, sample.
    def sample(distribution):
        return distribution.sample()
    # Create a version of f that uses this standard sampler
    runnable_f = inject_variables({'sample': sample}, f)
    # Run it
    return runnable_f(*args)

  
# Scoring
def sample_and_score(f):
    # First step is to create a version of f
    # that has non-standard scoring semantics.
    probability = 0
    all_samples = []
    def sampler_for_f(distribution):
        nonlocal probability
        x = distribution.sample()
        all_samples.append(x)
        probability += distribution.score(x)
        return x
    special_f = inject_variables({'sample': sampler_for_f}, f)

    # We now run the version of f that scores things,
    # and return the accumulated probability.
    retval = special_f()
    return retval, all_samples, probability