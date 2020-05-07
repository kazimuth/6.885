# Model
def f():
    p = sample(Uniform(), name='p')
    xs = []
    for i in range(10):
        xs.append(sample(Bernoulli(p), name=f'x{i}'))
    return sum(xs)

# Helper -- log categorical
def choose_particle(particles):
    weights = np.array([p['score'] for p in particles])
    return particles[np.random.choice(len(weights), p=np.exp(weights - logsumexp(weights)))]

# Importance resampler
def do_inference(f, observations, n_particles, args=[]):
    choices = {}
    score = 0
    def propose(distribution, name):
        nonlocal score
        if name in observations:
            score += distribution.score(observations[name])
            choices[name] = observations[name]
        else:
            choices[name] = distribution.sample()
        return choices[name]
    
    special_f = inject_variables({'sample': propose}, f)
    particles = []
    for i in range(n_particles):
        score = 0
        choices = {}
        retval = special_f(*args)
        particles.append({'score': score, 'choices': choices, 'retval': retval})
    return choose_particle(particles)

print(do_inference(f, {f'x{i}': True for i in range(10)}, 100))
