# MODEL

@compile_generative_function
def make_random_line():
    slope = sample(Normal(0, 2), "slope")
    intercept = sample(Normal(0, 2), "intercept")
    return lambda x: slope * x + intercept

@compile_generative_function
def add_noise(f):
    prob_outlier = sample(Uniform(), "prob_outlier")
    inlier_noise = sample(Uniform(), "inlier_noise")
    @compile_generative_function
    def noisy_line(x):
        if sample(Bernoulli(prob_outlier), "is_outlier?"):
            return sample(Normal(0, 10), "y")
        else:
            return sample(Normal(f(x), inlier_noise), "y")
    return noisy_line        

@compile_generative_function
def linear_regression_model(xs):
    random_line = call(make_random_line, "underlying_line")
    noisy_curve = call(add_noise, "add_noise", args=[random_line])
    ys = []
    for (i, x) in enumerate(xs):
        ys.append(call(noisy_curve, i, args=[x]))
    return ys
  

def mh(model, proposal, current_trace, args=[]):
    # Get proposed trace
    proposal_choices = proposal.generate([current_trace['choices'], args], {})['choices']
    
    # Get new trace
    proposed_choices = merge(current_trace['choices'], proposal_choices)
    new_trace = model.generate(args, proposed_choices)

    # Get q scores, forward and reverse
    forward_score = proposal.generate([current_trace['choices'], args], new_trace['choices'])['score']
    reverse_score = proposal.generate([new_trace['choices'], args], current_trace['choices'])['score']

    # Accept or reject
    log_accept_prob = new_trace['score'] - current_trace['score'] + reverse_score - forward_score

    if math.log(random.random()) < log_accept_prob:
        return new_trace
    else:
        return current_trace
      

@compile_generative_function
def line_proposal(current_choices, args):
    @compile_generative_function
    def helper():
        sample(Normal(current_choices['underlying_line']['slope'], 0.3), "slope")
        sample(Normal(current_choices['underlying_line']['intercept'], 0.3), "intercept")
    call(helper, "underlying_line")

@compile_generative_function
def noise_proposal(current_choices, args):
    @compile_generative_function
    def helper():
        sample(Uniform(), "prob_outlier")
        sample(Uniform(), "inlier_noise")
    call(helper, "add_noise")

def outlier_proposal(i):
    @compile_generative_function
    def propose_outlier(current_choices, args):
        @compile_generative_function
        def helper():
            sample(Bernoulli(0.01 if current_choices[i]["is_outlier?"] else 0.99), "is_outlier?")
        call(helper, i)
    return propose_outlier

def line_mh(xs, ys):
    obs = {i: {"y": ys[i]} for i in range(len(xs))}
    tr = linear_regression_model.generate([xs], obs)
    tr = linear_regression_model.generate([xs], tr['choices'])
    for i in range(1000):
        tr = mh(linear_regression_model, line_proposal, tr, args=[xs])
        tr = mh(linear_regression_model, noise_proposal, tr, args=[xs])
        for j in range(len(xs)):
            tr = mh(linear_regression_model, outlier_proposal(j), tr, args=[xs])
    return tr

print(line_mh(range(10), [2*i+2 for i in range(10)]))