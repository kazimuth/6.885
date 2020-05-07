# This doesn't work! Can you figure out how to make it work?

class PythonCodeGenerativeFunction(GenerativeFunction):
    def __init__(self, f):
        self.f = f
    
    def __call__(self, *args):
        def regular_sample(distribution, name):
            return distribution.sample()
        def regular_call(other_fn, name, args):
            return other_fn(*args)

        return inject_variables({'sample': regular_sample, 'call': regular_call}, self.f)(*args)
    
    # This line is new:
    @compile_generative_function
    def generate(self, args, observations={}):
        choices = {}
        score = 0

        def propose(distribution, name):
            nonlocal score
            if name in observations:
                choices[name] = observations[name]
                score += distribution.score(choices[name])
            else:
                choices[name] = sample(distribution, name) # Now, we use `sample` ourselves
            return choices[name]
        def call_for_generate(other_fn, namespace, args=[]):
            nonlocal score
            subobs = observations[namespace] if namespace in observations else {}
            result = call(other_fn.generate, namespace, args=[args, subobs])
            choices[namespace] = result['choices']
            score += result['score']
            return result['retval']
        
        retval = inject_variables({'sample': propose, 'call': call_for_generate}, self.f)(*args)
        return {'retval': retval, 'choices': choices, 'score': score}