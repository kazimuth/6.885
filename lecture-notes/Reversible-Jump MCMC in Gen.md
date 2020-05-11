# Problem Set 2C: Reversible-Jump MCMC in Gen _(with applications to Program Synthesis)_

### What is this notebook about?

In the previous problem set, we saw how to write custom Metropolis-Hastings proposals and stitch them together to craft our own MCMC algorithms. Recall that to write a custom MH proposal, it was necessary to define a generative function that accepted as an argument the previous trace, and proposed a new trace *by "acting out" the model*, that is, by sampling new proposed values of random choices _at the same addresses used by the model_.

For example, a random walk proposal for the addresses `:x` and `:y` might have looked like this:

```julia
@gen function random_walk_proposal(previous_trace)
    x ~ normal(previous_trace[:x], 0.1)
    y ~ normal(previous_trace[:y], 0.2)
end
```

This pattern implies a severe restriction on the kinds of proposals we can write: the new `x` and `y` values must be sampled directly from a Ggen distribution (e.g., `normal`). For example, it is impossible to use this pattern to define a proposal that deterministcally *swaps* the current `x` and `y` values.

In this notebook, you'll learn a more flexible approach to defining custom MH proposals. The technique is widely applicable, but is particularly well-suited to models with discrete and continuous parameters, where the discrete parameters determine _which_ continuous parameters exist.

The polynomial curve-fitting demo we saw in the last problem set is one simple example: the discrete parameter `degree` determines which coefficient parameters exist. This situation is also common in program synthesis: discrete variables determine a program's structure, but the values inside the program may be continuous.



## Outline

**Section 1.** [Recap of the piecewise-function model](#piecewise-constant)

**Section 2.** [Basic Metropolis-Hastings inference](#basic-mh)

**Section 3.** [Reversilbe-Jump "Split-Merge" proposals](#split-merge)

**Section 4.** [Bayesian program synthesis of GP kernels](#synthesis)

**Section 5.** [A tree regeneration proposal](#tree-regen)


```julia
using Gen
using GenViz
include("dirichlet.jl")

# Do not run this cell more than once!
server = VizServer(8090);
```

## 1. Recap of the piecewise-constant function model  <a name="piecewise-constant"></a>

In Problem Set 1A, you worked with a model of _piecewise constant_ functions, with unknown changepoints. Here, we model the same scenario, but somewhat differently.

Given a dataset of `xs`, our model will randomly divide the range `(xmin, xmax)` into a random number of segments.

It does this by sampling a number of segments (`:segment_count`), then sampling a vector of _proportions_ from a Dirichlet distribution (`:fractions`). The vector is guaranteed to sum to 1: if there are, say, three segments, this vector might be `[0.3, 0.5, 0.2]`. The length of each segment is the fraction of the interval assigned to it, times the length of the entire interval, e.g. `0.2 * (xmax - xmin)`. For each segmment, we generate a `y` value from a normal distribution. Finally, we sample the `y` values near the piecewise constant function described by the segments.

### Using `@dist` to define new distributions for convenience
To sample the number of segments, we need a distribution with support only on the positive integers. We create one using the [Distributions DSL](https://probcomp.github.io/Gen/dev/ref/distributions/#dist_dsl-1):


```julia
# A distribution that is guaranteed to be 1 or higher.
@dist poisson_plus_one(rate) = poisson(rate) + 1;
```

Distributions declared with `@dist` can be used to make random choices inside of Gen models.
Behind the scenes, `@dist` has analyzed the code and figured out how to evaluate the `logpdf`,
or log density, of our newly defined distribution. So we can ask, e.g., what the density of
`poisson_plus_one(1)` is at the point `3`:




```julia
logpdf(poisson_plus_one, 3, 1)
```




    -1.6931471805599454



Note that this is the same as the logpdf of `poisson(1)` at the point `2` — `@dist`'s main job is to
automate the logic of converting the above call into this one:


```julia
logpdf(poisson, 2, 1)
```




    -1.6931471805599454



### Writing the model

We can now write the model itself. It is relatively straightforward, though there
are a few things you may not have seen before:

* *List comprehensions* allow you to create a list without writing an entire
  for loop. For example, `[{(:segments, i)} ~ normal(0, 1) for i=1:segment_count]`
  creates a list of elements, one for each `i` in the range `[1, ..., segment_count]`,
  each of which is generated using the expression `{(:segments, i)} ~ normal(0, 1)`.

* The [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution) is
  a distribution over the *simplex* of vectors whose elements sum to 1. We use it to
  generate the fractions of the entire interval (x_min, x_max) that each segment of our
  piecewise function covers.

* The logic to generate the `y` points is as follows: we compute the cumulative
  fractions `cumfracs = cumsum(fractions)`, such that `xmin + (xmax - xmin) * cumfracs[j]` is
  the x-value of the right endpoint of the `j`th segment.  Then we sample at each
  address `(:y, i)` a normal whose mean is the y-value of the segment that
  contains `xs[i]`.


```julia
@gen function piecewise_constant(xs::Vector{Float64})
    # Generate a number of segments (at least 1)
    segment_count ~ poisson_plus_one(1)

    # To determine changepoints, draw a vector on the simplex from a Dirichlet
    # distribution. This gives us the proportions of the entire interval that each
    # segment takes up. (The entire interval is determined by the minimum and maximum
    # x values.)
    fractions ~ dirichlet([1.0 for i=1:segment_count])

    # Generate values for each segment
    segments = [{(:segments, i)} ~ normal(0, 1) for i=1:segment_count]

    # Determine a global noise level
    noise ~ gamma(1, 1)

    # Generate the y points for the input x points
    xmin, xmax = extrema(xs)
    cumfracs = cumsum(fractions)
    # Correct numeric issue: `cumfracs[end]` might be 0.999999
    @assert cumfracs[end] ≈ 1.0
    cumfracs[end] = 1.0

    inds = [findfirst(frac -> frac >= (x - xmin) / (xmax - xmin),
                      cumfracs) for x in xs]
    segment_values = segments[inds]
    for (i, val) in enumerate(segment_values)
        {(:y, i)} ~ normal(val, noise)
    end
end;
```

Let's understand its behavior by visualizing several runs of the model.
We begin by creating a simple dataset of xs, evenly spaced between -5
and 5.


```julia
xs_dense = collect(range(-5, stop=5, length=50));
```

Don't worry about understanding the following code, which we use for visualization.


```julia
function trace_to_dict(tr)
    Dict("values" => [tr[(:segments, i)] for i=1:(tr[:segment_count])],
         "fracs"  => tr[:fractions], "n" => tr[:segment_count], "noise" => tr[:noise],
         "y-coords" => [tr[(:y, i)] for i=1:length(xs_dense)])
end;
```


```julia
viz = Viz(server, joinpath(@__DIR__, "piecewise-constant-viz/dist"), [xs_dense])
for i=1:10
    tr = simulate(piecewise_constant, (xs_dense,))
    putTrace!(viz, "t$(i)", trace_to_dict(tr))
end
displayInNotebook(viz)
```



Many of the samples involve only one segment, but many of them involve more. The level of noise also varies from sample to sample.

## 2. Basic Metropolis-Hastings inference <a name="basic-mh" />

Let's create three synthetic datasets, each more challenging than the last, to test out our inference capabilities.


```julia
ys_simple  = ones(length(xs_dense)) .+ randn(length(xs_dense)) * 0.1
ys_medium  = Base.ifelse.(Int.(floor.(abs.(xs_dense ./ 3))) .% 2 .== 0,
                          2, 0) .+ randn(length(xs_dense)) * 0.1;
ys_complex = Int.(floor.(abs.(xs_dense ./ 2))) .% 5 .+ randn(length(xs_dense)) * 0.1;
```

We'll need a helper function for creating a choicemap of constraints from a vector of `ys`:


```julia
function make_constraints(ys)
    choicemap([(:y, i) => ys[i] for i=1:length(ys)]...)
end;
```

As we saw in the last problem set, importance sampling does a decent job on the simple dataset:


```julia
NUM_CHAINS = 10

viz = Viz(server, joinpath(@__DIR__, "piecewise-constant-viz/dist"), [xs_dense])
for i=1:NUM_CHAINS
    # Perform importance sampling with 5000 particles
    (tr, _) = importance_resampling(piecewise_constant, (xs_dense,), make_constraints(ys_simple), 5000)
    putTrace!(viz, "t$(i)", trace_to_dict(tr))
end
displayInNotebook(viz)
```


But on the complex dataset, it takes many more particles (here, we use 50,000) to do even an OK job:


```julia
viz = Viz(server, joinpath(@__DIR__, "piecewise-constant-viz/dist"), [xs_dense])
scores = Vector{Float64}(undef, NUM_CHAINS)
for i=1:NUM_CHAINS
    println("Inferring $(i)/$(NUM_CHAINS) traces...")
    (tr, _) = importance_resampling(piecewise_constant, (xs_dense,), make_constraints(ys_complex), 50000)
    scores[i] = get_score(tr)
    putTrace!(viz, "t$(i)", trace_to_dict(tr))
end
displayInNotebook(viz)

println("Log mean score: $(logsumexp(scores)-log(NUM_CHAINS))")
```


    Log mean score: -44.03537745893023


Let's try instead to write a simple Metropolis-Hastings algorithm to tackle the problem.

First, some visualization code (feel free to ignore it!).


```julia
# GenViz visualization for an MH algorithm on the piecewise-constant problem.
function visualize_mh_alg(xs, ys, update, iters=2000, N=NUM_CHAINS)
    viz = Viz(server, joinpath(@__DIR__, "piecewise-constant-viz/dist"), [xs])
    scores = displayInNotebook(viz) do
        scores = Float64[]
        for i=1:N
            tr, = generate(piecewise_constant, (xs,), make_constraints(ys))
            for iter=1:iters
                tr = update(tr, xs, ys)
                putTrace!(viz, "t$i", trace_to_dict(tr))
            end
            push!(scores, get_score(tr))
        end
        scores
    end

    println(logsumexp(scores) - log(N))
end
```




    visualize_mh_alg (generic function with 3 methods)



And now the MH algorithm itself.

We'll use a basic Block Resimulation sampler, which cycles through the following blocks of variables:
* Block 1: `:segment_count` and `:fractions`. Resampling these tries proposing a completely
  new division of the interval into pieces. However, it reuses the `(:segments, i)` values
  wherever possible; that is, if we currently have three segments and `:segment_count` is proposed
  to change to 5, only two new segment values will be sampled.

* Block 2: `:fractions`. This proposal tries leaving the number of segments the same, but
  resamples their relative lengths.

* Block 3: `:noise`. This proposal adjusts the global noise parameter.

* Blocks 4 and up: `(:segments, i)`. Tries separately proposing new values for each segment (and accepts
  or rejects each proposal independently).


```julia
function simple_update(tr, xs, ys)
    tr, = mh(tr, select(:segment_count, :fractions))
    tr, = mh(tr, select(:fractions))
    tr, = mh(tr, select(:noise))
    for i=1:tr[:segment_count]
        tr, = mh(tr, select((:segments, i)))
    end
    tr
end
```




    simple_update (generic function with 1 method)



Our algorithm makes quick work of the simple dataset:


```julia
visualize_mh_alg(xs_dense, ys_simple, simple_update, 500)
```


    45.525284680963686


On the medium dataset, it does an OK job with less computation than we used in importance sampling:


```julia
visualize_mh_alg(xs_dense, ys_medium, simple_update, 2000)
```



    27.601494775638955


But on the complex dataset, it is still unreliable:


```julia
visualize_mh_alg(xs_dense, ys_complex, simple_update, 2000)
```

    -24.72370786333229


### Problem 2.9: Smarter Metropolis-Hastings

One problem with the simple block resimulation algorithm is that the proposals for `(:segments, i)` are
totally uninformed by the data. In this problem, you'll write a custom proposal (using the techniques we
covered in Problem Set 1B) that uses the data to propose good values of `y` for each segment.

Write a generative function `segments_proposal` that can serve as a smart proposal distribution for
this problem. It should:

* Propose a new `:segment_count` from `poisson_plus_one(1)` (the prior).
* Propose a new `:fractions` from `dirichlet([1.0 for i=1:segment_count])` (the prior).
* In each segment, propose the function value `(:segments, i)` to be (a noisy version of) the
  average of the `y` values in our dataset from the given segment. Draw `(:segments, i)` from
  a normal distribution with that mean, and a small standard deviation (e.g. 0.3).

We will use this proposal to replace the "Block 1" move from our previous algorithm. This should
make it easier to have proposals accepted, because whenever we propose a new segmentation of the
interval, we propose it with reasonable `y` values attached.

### Solution 2.9

We have provided some starter code:


```julia
@gen function segments_proposal(t, xs, ys)
    x_min, x_max = minimum(xs), maximum(xs)
    x_range = x_max - x_min

    segment_count ~ poisson_plus_one(1)
    fractions ~ dirichlet([1.0 for i=1:segment_count])

    start = x_min

    for i=1:segment_count
        sum = 0.0
        count = 0

        finish = start + fractions[i] * x_range

        for (x, y) in zip(xs, ys)
            if start <= x && x < finish
                sum += y
                count += 1
            end
        end

        avg = if count > 0
            sum/count
        else
            0
        end

        {(:segments, i)} ~ normal(avg, 0.3)

        start = finish
    end
end;
```

We define `custom_update` to use `segments_proposal` in place of the first block from `simple_update`.


```julia
function custom_update(tr, xs, ys)
    (tr, _) = mh(tr, segments_proposal, (xs, ys))
    (tr, _) = mh(tr, select(:fractions))
    (tr, _) = mh(tr, select(:noise))
    for i=1:(tr[:segment_count])
        (tr, _) = mh(tr, select((:segments, i)))
    end
    tr
end
```




    custom_update (generic function with 1 method)



Let's see how this one does on each dataset:


```julia
visualize_mh_alg(xs_dense, ys_medium, custom_update, 2000)
```



    28.266353177085076



```julia
visualize_mh_alg(xs_dense, ys_complex, custom_update, 2000)
```



    -2.96042589845042


This will often outperform the simplest MH solution, but still leaves something to be desired. The smart `segment_proposal` helps find good function values for each segment, but doesn't help us in cases where the segment proportions are wrong; in these cases, the model just decides that noise must be high.

## 3. Involution MH for Split-Merge proposals <a name="split-merge" />

How might we improve on the MH algorithms from the previous section, to more effectively search for good values of the `fractions` variable?

One approach is to add new proposals to the mix that _iteratively refine_ the `fractions`, rather than
relying on blind resimulation. A natural strategy might be to add _split_ and _merge_ proposals:

* A _split_ chooses a segment to break into two pieces at a random point (and chooses new values for the two segments).
* A _merge_ chooses two adjacent segments to merge together into one segment (with a shared value).


### MCMC, Metropolis-Hastings, and Reversibility
Note that alone, neither of these two proposals makes for a valid Metropolis-Hastings transition proposal.
Why? One requirement of MCMC algorithms is that each transition kernel (kernel, not proposal: **in MH,
the kernel includes the accept-reject step**) _leave the posterior distribution stationary_.
What does this mean? Suppose we somehow obtained an oracle that let us sample a trace from the exact posterior,
$t \sim p(t \mid \text{observations})$. Then if we run an MCMC transition kernel $T$ on the
trace to obtain a new trace $t' \sim T(t' \leftarrow t)$, the marginal distribution of $t'$ should still
be the posterior. In simpler terms, no one should be able to tell the difference between "traces sampled
from the posterior" (i.e., using the output of the oracle directly) and "traces sampled from the posterior
then sent through a transition kernel" (i.e., passing the output of the oracle as the input to the transition
kernel and using the result).
As a formula,

$$\int p(t \mid \text{observations}) \, T(t' \leftarrow t) \, \mathrm{d}t = p(t' \mid \text{observations})$$

Now suppose we set our transition kernel $T$ to a Metropolis-Hastings `split` move. The `split` move
_always_ increases the number of segments. So if `t` comes from the true posterior, then the distribution
of `t'`, in terms of number of segments, will necessarily be shifted upward (unless MH deterministically
_rejects_ every proposal -- which is what MH in Gen will do if given a `split` proposal). The same issue exists,
in reverse, for the `merge` proposal.

In general, this "stationarity" rule means that our Metropolis-Hastings proposals must be
*reversible*, meaning that if there is some probability that a proposal can take you
from one region of the state space to another, it must also have some probability of sending
you back from the new region to the old region. If this criterion is satisfied, then the MH
accept-reject step can accept and reject proposals with the proper probabilities to ensure that
the stationarity property described above holds.

To make "split" and "merge" fulfill this "reversibility" criterion, we can think of them as a
constituting a *single* proposal, which *randomly chooses* whether to split or merge at each iteration.
This is an example of a _reversible-jump_ proposal [1].

**References:**
1. Green, Peter J., and David I. Hastie. "Reversible jump MCMC." Genetics 155.3 (2009): 1391-1403.

### Implementing Split-Merge in Gen
This is a sensible proposal. But if we try to write this proposal in Gen, we quickly hit several roadblocks. For example:

* The proposal needs to make several random choices that are *not* meant to serve as
  proposals for corresponding random choices in the model. For example, the proposal
  must decide whether to "split" or "merge," and then it needs to decide _at which index_ it
  will split or merge. But Gen interprets every traced random choice made by a proposal
  as corresponding to some choice in the model.
* Once we choose to split (or merge), it's unclear how we should propose to the various
  relevant addresses: from what distribution should the proposal sample `fractions`, for
  example? What we need is to propose a _deterministic_ value for `fractions`, _based on_
  the random choices the proposal makes.

To get around these issues, Gen provides a variant of Metropolis-Hastings that is a bit trickier to use but is ultimately more flexible.


The idea is this: first, we write a generative function that samples all the randomness
the proposal will require. In our case, this will involve

* choosing whether to split or merge
* choosing at what index the split or merge will happen
* if splitting, choosing where in a segment to split
* choosing new values for merged segments or newly created split segments


```julia
@gen function split_merge_proposal_randomness(t)
    old_n = t[:segment_count]

    # Choose whether to split (T) or to merge (F), keeping in mind
    # that if old_n == 1, then our decision is made for us
    # (split).
    if ({:split_or_merge} ~ bernoulli(old_n == 1 ? 1 : 0.3))

        # split
        # What index to split at?
        index ~ uniform_discrete(1,old_n)
        # Where is the splitting point, relative to the segment being split?
        split_percentage ~ uniform(0, 1)
        # New values for the two new segments
        new_value_1 ~ normal(t[(:segments, index)], 0.1)
        new_value_2 ~ normal(t[(:segments, index)], 0.1)
    else
        # merge
        # What index to merge at? (Merge index i and i + 1)
        index ~ uniform_discrete(1, old_n-1) # merge i and i+1

        # Sample a new value for the merged segment, near the mean of the
        # two existing segments.
        new_value ~ normal((t[(:segments, index)] + t[(:segments, index+1)]) / 2.0, 0.1)
    end
end;
```

Let's look at what this function samples:


```julia
tr, = generate(piecewise_constant, (xs_dense,), make_constraints(ys_complex,));
println(get_choices(simulate(split_merge_proposal_randomness, (tr,))))
```

    │
    ├── :index : 2
    │
    ├── :split_or_merge : false
    │
    └── :new_value : -1.003804961645505



We now need to write an ordinary Julia function that tells Gen how to use
the proposal _randomness_ generated by the generative function above to create
a proposed next trace for Metropolis-Hastings.

The function takes four **inputs**:

* The previous trace `t`
* The proposal randomness generated by our generative function above, `forward_randomness`
* The return value of the generative function we defined, `forward_ret` -- we won't use this for now
* The arguments, other than `t`, that were used to generate `forward_randomness` from the generative function written above. In our case, that function has no additional arguments, so this will always be empty.

It is supposed to return three **outputs**:

* `new_trace`: an updated trace to propose. We can construct this however we want based on `t` and `forward_choices`.
* `backward_choices`: a choicemap for the `proposal_randomness` generative function, capable of "sending `new_trace` back to the old trace `t`". For example, if we are enacting a `split` proposal, this would specify the precise `merge` proposal necessary to undo the split. This serves as "proof" that our proposal really is reversible.
* `weight`: usually, `get_score(new_trace) - get_score(t)`. The reason we need to return this is that it is possible to write proposals that need to return something different here; if your proposal involves deterministically transforming continuous random variables in non-volume-preserving ways, see the Gen documentation for details on how `weight` should change. In this notebook, `weight` will always be `get_score(new_trace) - get_score(t)`.

We call the Julia function an *involution*, because it must have the following property.
Suppose we propose a new trace using `proposal_randomness_func` and then our Julia function
`f`:

```
forward_choices = get_choices(simulate(proposal_randomness_func, (old_trace,)))
new_trace, backward_choices, = f(old_trace, forward_choices, ...)
```

Now suppose we use `backward_choices` and ask what `f` would do to `new_trace`:

```
back_to_the_first_trace_maybe, backward_backward_choices, = f(new_trace, backward_choices, ...)
```

Then we require that `backward_backward_choices == forward_choices`, and that `back_to_the_first_trace_maybe == old_trace`.

Here's an involution for Split-Merge:


```julia
function involution(t, forward_choices, forward_retval, proposal_args)
    # Goal: based on `forward_choices`, return a new trace
    # and the `backward_choices` required to reverse this
    # proposal.
    new_trace_choices = choicemap()
    backward_choices  = choicemap()

    # First, check whether we're dealing with a split or a merge.
    split_or_merge = forward_choices[:split_or_merge]

    # To reverse a split, use a merge (and vice versa)
    backward_choices[:split_or_merge] = !split_or_merge

    # Where is the split / merge occurring?
    # Reversing the proposal would use the same index.
    index = forward_choices[:index]
    backward_choices[:index] = index

    # Now we handle the segment values and proportions.
    # First, pull out the existing values from the previous
    # trace `t`. IMPORTANT NOTE: we need to `copy`
    # `t[:fractions]`, because we intend to perform mutating
    # Julia operations like `insert!` and `deleteat!` on it,
    # but do not wish to change the memory underlying the
    # original trace `t` (in case the proposal is rejected
    # and we need to return to the original trace).
    fractions      = copy(t[:fractions])
    segment_values = [t[(:segments, i)] for i=1:t[:segment_count]]

    # How we update `fractions` and `segment_values` depends on
    # whether this is a split or a merge.
    if split_or_merge
        # If this is a split, then add a new element at `index`,
        # according to the split proportion.
        insert!(fractions, index, fractions[index] * forward_choices[:split_percentage])
        fractions[index + 1] *= (1 - forward_choices[:split_percentage])

        # Segment values
        backward_choices[:new_value] = segment_values[index]
        segment_values[index] = forward_choices[:new_value_1]
        insert!(segment_values, index, forward_choices[:new_value_2])
    else
        # If this is a merge, then combine the two segments `index`
        # and `index + 1`.
        proportion = fractions[index] / (fractions[index] + fractions[index + 1])
        fractions[index] += fractions[index + 1]
        deleteat!(fractions, index + 1)

        # Set the relevant segment values.
        backward_choices[:new_value_1] = segment_values[index]
        backward_choices[:new_value_2] = segment_values[index + 1]
        backward_choices[:split_percentage] = proportion
        segment_values[index] = forward_choices[:new_value]
        deleteat!(segment_values, index + 1)
    end

    # Fill a choicemap of the newly proposed trace's values
    new_trace_choices[:fractions] = fractions
    for (i, value) in enumerate(segment_values)
        new_trace_choices[(:segments, i)] = value
    end
    new_trace_choices[:segment_count] = length(fractions)

    # Obtain an updated trace matching the choicemap, and a weight
    new_trace, weight, = update(t, get_args(t), (NoChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
end
```




    involution (generic function with 1 method)



Look at the last two lines of the involution. The involution is responsible for returning a
*new trace*, not just a new *choicemap*. So, the common pattern to follow in involutions is:

* Throughout the involution, fill a choicemap (here, `new_trace_choices`) with the updates
  that you want to make to the old trace `t`.
* At the end, call Gen's `update` function, passing in (1) the old trace `t`, (2) the model
  function's arguments, (3) a tuple of "argdiffs" (in our case, we know that the argument `xs`
  has not changed, so we pass in `NoChange()`), and (4) the choicemap of updates. `update` returns
  two useful values: the new trace, and a `weight`, which is equal to `get_score(t') - get_score(t)`.
  This is the weight we need to return from the involution.


We can create a new MH update that uses our proposal, and test it on the new dataset:


```julia
@gen function mean_segments_proposal(t, xs, ys, i)
    xmin, xmax = minimum(xs), maximum(xs)
    x_range = xmax - xmin
    fracs = t[:fractions]
    min = xmin + x_range * sum(fracs[1:i-1])
    max = xmin + x_range * sum(fracs[1:i])
    relevant_ys = [y for (x,y) in zip(xs,ys) if x >= min && x <= max]
    {(:segments, i)} ~ normal(sum(relevant_ys)/length(relevant_ys), 0.3)
end

function custom_update_inv(tr, xs, ys)
    tr, accepted = mh(tr, split_merge_proposal_randomness, (), involution)
    for i=1:tr[:segment_count]
        tr, = mh(tr, mean_segments_proposal, (xs, ys, i))
    end
    tr, = mh(tr, select(:noise))
    tr
end

visualize_mh_alg(xs_dense, ys_complex, custom_update_inv, 2000)
```


body, html { margin-left: 0px; margin-right: 0px; }
h1[data-v-678d3a0d] { font-family: Avenir, Helvetica, Arial, sans-serif; text-align: center; }
#traces[data-v-678d3a0d] { display: flex; -moz-box-orient: horizontal; -moz-box-direction: normal; flex-flow: row wrap; }
#app { font-family: Avenir, Helvetica, Arial, sans-serif; color: rgb(44, 62, 80); }</style>


    36.32939987016745


## Exercise 2.10

Here are some possible "improvements" to the above split-merge proposal, which are all in fact _invalid_.
For each, explain why it is invalid, and briefly describe a version of the suggested improvement that _would_
be valid.

1. Proposed improvement: make the `split` smarter, by always setting `split_proportion` to split
   somewhere between points `i` and `i+1`, where `i` and `i+1` are the two adjacent datapoints that have
   the largest $|y_i - y_{i+1}|$ value. The precise split proportion can still be random, drawn so that
   the split occurs uniformly between $x_i$ and $x_{i+1}$

2. Proposed improvement: when choosing an index to split or merge, make smart choices: when splitting,
   choose the segment that currently explains its data least well (according to the likelihood of the `y`
   points that fall on the segment), and when merging, choose the two adjacent segments whose existing
   values are closest to one another.

3. Proposed improvement: when splitting, do not sample `:new_value_1` and `:new_value_2` randomly; instead,
   in the involution, set them equal to the mean values of the datapoints that fall within their segment
   intervals.



## Solution 2.10

Please answer below:

"Improvement" 1:

**Why it's invalid:** It will always propose splitting at the same (approximate) location, while merges can occur anywhere. Therefore it won't create a stationary kernel: if there is a split outside in the range $[x_i, x_{i+1}]$, which is merged, that split can never be created again by this kernel!

**Alternative approach:**
Sample the index to split on, with points with larger $\Delta y$ values weighted more highly. Keep the random proportion choice how it was.

"Improvement" 2:

**Why it's invalid:** It can't merge segments that have distant values, so it will never (?) sample traces with bad merges, which means that it won't describe the whole probability space. (I interpret "existing values" to mean "x values within the segment".)

**Alternative approach:** Sample split / merge choices with odds proportional to the given criteria.

(Also, if you instead chose to merge based on the values of segments rather than the observations covered by those segments, I *think* that might be valid too? Since you could propose bad segments close to existing ones, you could get anywhere in the space. Not sure though.)

"Improvement" 3:

**Why it's invalid:** It makes it impossible for the kernel to sample splits with more incorrect values, which are still possible according to the model.

**Alternative approach:**
Sample normal distributions centered on the means.


## 4. Bayesian synthesis of Gaussian Process kernel programs  <a name="synthesis"></a>

We now consider a more complex model, adapted from the [Bayesian synthesis paper](https://popl19.sigplan.org/details/POPL-2019-Research-Papers/79/Bayesian-Synthesis-of-Probabilistic-Programs-for-Automatic-Data-Modeling) by Saad et al. that you saw in lecture.

Our goal is to analyze time-series data: for some sequence of times $t_1, t_2, \dots, t_n$, we have observed points $x(t_1), x(t_2), \dots, x(t_n)$. We'll model a time series as arising from a Gaussian process with a particular _covariance kernel_. A covariance kernel is a function that takes in two times $t_1$ and $t_2$, and
computes a covariance between the random variables $x(t_1)$ and $x(t_2)$. Different covariance kernels can encode various structured patterns in the data. For example:

* A _squared exponential_ kernel assigns high covariance to pairs of time points that are nearby, and lower covariance to pairs of time points that are far away.  This kernel encodes a _smoothness_ prior: the time series does not change drastically from one moment to the next.

* A _periodic_ kernel assigns a covariance that depends on $\sin(f \cdot |t_1 - t_2|)$ for some frequency `f`. This allows it to assign high covariance to points that lie the "right" distance from one another, but not too near or too far, encoding a periodic structure: the time series is highly correlated with itself, some time earlier.

* A _constant_ kernel encodes some constant covariance between $x(t_1)$ and $x(t_2)$, no matter what $t_1$ and $t_2$ are.

These primitive kernels can be composed in various ways, e.g., addition and multiplication, to encode more complex features of time series.

The main inference task we address in this section will be to infer a covariance kernel that captures the structure of a time series. In particular, given a dataset of $t_i$, $x(t_i)$ pairs, we would like to infer a covariance kernel `k` that makes the dataset likely under $GP(k)$, a Gaussian process with $k$ as a covariance kernel.

Covariance kernels will be represented by _expressions_ in a simple domain-specific language (DSL). The simplest of these expressions are primitive kernels, each of which comes with one or more parameters:

* `Constant(param :: Float64)`
* `Linear(param :: Float64)`
* `SquaredExponential(length_scale :: Float64)`
* `Periodic(scale :: Float64, period :: Float64)`

Smaller expressions can be combined into larger ones using _binary operations_ for addition and multiplication:

* `Plus(left, right)`
* `Times(left, right)`

Here, `left` and `right` are _other_ covariance kernel expressions.

An expression, like `Plus(Constant(0.3), Periodic(3.2, 1.9))`, can be viewed as a _syntax tree_.

Primitives form the leaf nodes of the tree; binary operations form the internal nodes, each of which has a `left` and `right` subtree. The code below defines this tree datatype for representing expressions, which we call `Kernel`.


```julia
import LinearAlgebra

"""Node in a tree representing a covariance function"""
abstract type Kernel end
abstract type PrimitiveKernel <: Kernel end
abstract type CompositeKernel <: Kernel end

"""
    size(::Kernel)
Number of nodes in the tree describing this kernel.
"""
Base.size(::PrimitiveKernel) = 1
Base.size(node::CompositeKernel) = node.size

"""Constant kernel"""
struct Constant <: PrimitiveKernel
    param::Float64
end


eval_cov(node::Constant, x1, x2) = node.param

function eval_cov_mat(node::Constant, xs::Vector{Float64})
    n = length(xs)
    fill(node.param, (n, n))
end


"""Linear kernel"""
struct Linear <: PrimitiveKernel
    param::Float64
end


eval_cov(node::Linear, x1, x2) = (x1 - node.param) * (x2 - node.param)

function eval_cov_mat(node::Linear, xs::Vector{Float64})
    xs_minus_param = xs .- node.param
    xs_minus_param * xs_minus_param'
end


"""Squared exponential kernel"""
struct SquaredExponential <: PrimitiveKernel
    length_scale::Float64
end


eval_cov(node::SquaredExponential, x1, x2) =
    exp(-0.5 * (x1 - x2) * (x1 - x2) / node.length_scale)

function eval_cov_mat(node::SquaredExponential, xs::Vector{Float64})
    diff = xs .- xs'
    exp.(-0.5 .* diff .* diff ./ node.length_scale)
end


"""Periodic kernel"""
struct Periodic <: PrimitiveKernel
    scale::Float64
    period::Float64
end


function eval_cov(node::Periodic, x1, x2)
    freq = 2 * pi / node.period
    exp((-1/node.scale) * (sin(freq * abs(x1 - x2)))^2)
end


function eval_cov_mat(node::Periodic, xs::Vector{Float64})
    freq = 2 * pi / node.period
    abs_diff = abs.(xs .- xs')
    exp.((-1/node.scale) .* (sin.(freq .* abs_diff)).^2)
end


"""Plus node"""
struct Plus <: CompositeKernel
    left::Kernel
    right::Kernel
    size::Int
end


Plus(left, right) = Plus(left, right, size(left) + size(right) + 1)


function eval_cov(node::Plus, x1, x2)
    eval_cov(node.left, x1, x2) + eval_cov(node.right, x1, x2)
end


function eval_cov_mat(node::Plus, xs::Vector{Float64})
    eval_cov_mat(node.left, xs) .+ eval_cov_mat(node.right, xs)
end


"""Times node"""
struct Times <: CompositeKernel
    left::Kernel
    right::Kernel
    size::Int
end


Times(left, right) = Times(left, right, size(left) + size(right) + 1)


function eval_cov(node::Times, x1, x2)
    eval_cov(node.left, x1, x2) * eval_cov(node.right, x1, x2)
end


function eval_cov_mat(node::Times, xs::Vector{Float64})
    eval_cov_mat(node.left, xs) .* eval_cov_mat(node.right, xs)
end
```




    eval_cov_mat (generic function with 6 methods)



More helper functions for evaluating covariances:


```julia
"""Compute covariance matrix by evaluating function on each pair of inputs."""
function compute_cov_matrix(covariance_fn::Kernel, noise, xs)
    n = length(xs)
    cov_matrix = Matrix{Float64}(undef, n, n)
    for i=1:n
        for j=1:n
            cov_matrix[i, j] = eval_cov(covariance_fn, xs[i], xs[j])
        end
        cov_matrix[i, i] += noise
    end
    return cov_matrix
end


"""Compute covariance function by recursively computing covariance matrices."""
function compute_cov_matrix_vectorized(covariance_fn, noise, xs)
    n = length(xs)
    eval_cov_mat(covariance_fn, xs) + Matrix(noise * LinearAlgebra.I, n, n)
end

"""
Computes the conditional mean and covariance of a Gaussian process with prior mean zero
and prior covariance function `covariance_fn`, conditioned on noisy observations
`Normal(f(xs), noise * I) = ys`, evaluated at the points `new_xs`.
"""
function compute_predictive(covariance_fn::Kernel, noise::Float64,
                            xs::Vector{Float64}, ys::Vector{Float64},
                            new_xs::Vector{Float64})
    n_prev = length(xs)
    n_new = length(new_xs)
    means = zeros(n_prev + n_new)
    cov_matrix = compute_cov_matrix(covariance_fn, noise, vcat(xs, new_xs))
    cov_matrix_11 = cov_matrix[1:n_prev, 1:n_prev]
    cov_matrix_22 = cov_matrix[n_prev+1:n_prev+n_new, n_prev+1:n_prev+n_new]
    cov_matrix_12 = cov_matrix[1:n_prev, n_prev+1:n_prev+n_new]
    cov_matrix_21 = cov_matrix[n_prev+1:n_prev+n_new, 1:n_prev]
    @assert cov_matrix_12 == cov_matrix_21'
    mu1 = means[1:n_prev]
    mu2 = means[n_prev+1:n_prev+n_new]
    conditional_mu = mu2 + cov_matrix_21 * (cov_matrix_11 \ (ys - mu1))
    conditional_cov_matrix = cov_matrix_22 - cov_matrix_21 * (cov_matrix_11 \ cov_matrix_12)
    conditional_cov_matrix = 0.5 * conditional_cov_matrix + 0.5 * conditional_cov_matrix'
    (conditional_mu, conditional_cov_matrix)
end

"""
Predict output values for some new input values
"""
function predict_ys(covariance_fn::Kernel, noise::Float64,
                    xs::Vector{Float64}, ys::Vector{Float64},
                    new_xs::Vector{Float64})
    (conditional_mu, conditional_cov_matrix) = compute_predictive(
        covariance_fn, noise, xs, ys, new_xs)
    mvnormal(conditional_mu, conditional_cov_matrix)
end
```




    predict_ys



### A prior over kernel programs

Our goal will be to infer a covariance kernel program that explains our time series.
To do this, we begin by placing a prior over covariance kernels. The main ingredient
will be the following distribution over Kernel Types:


```julia
kernel_types = [Constant, Linear, SquaredExponential, Periodic, Plus, Times]
@dist choose_kernel_type() = kernel_types[categorical([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])];
```

It is essentially a categorical distribution, but one that outputs kernel types
instead of integers.

We then write a prior over kernels that uses `choose_kernel_type` recursively:

1. It first samples a parent kernel type using our `choose_kernel_type` distribution.
2. If the kernel is `Plus` or `Times`, it recursively calls itself to generate two sub-kernels
   (called `:left` and `:right`), and returns the composite kernel.
3. If the kernel is a primitive, then it samples the parameters from `uniform(0, 1)` distributions.
   (Note that `Periodic` kernels have two parameters, unlike the others, so we handle it specially.)


```julia
# Prior on kernels
@gen function covariance_prior()
    # Choose a type of kernel
    kernel_type ~ choose_kernel_type()

    # If this is a composite node, recursively generate subtrees
    if in(kernel_type, [Plus, Times])
        return kernel_type({:left} ~ covariance_prior(), {:right} ~ covariance_prior())
    end

    # Otherwise, generate parameters for the primitive kernel.
    kernel_args = (kernel_type == Periodic) ? [{:scale} ~ uniform(0, 1), {:period} ~ uniform(0, 1)] : [{:param} ~ uniform(0, 1)]
    return kernel_type(kernel_args...)
end
```




    DynamicDSLFunction{Any}(Dict{Symbol,Any}(), Dict{Symbol,Any}(), Type[], ##covariance_prior#429, Bool[], false)



Let's look at some sampled kernel programs from the model:


```julia
for i=1:10
    println(covariance_prior())
end
```

    Linear(0.35002273316040466)
    Times(Constant(0.5886563157056535), SquaredExponential(0.6742041986112906), 3)
    Periodic(0.49432484698379175, 0.3984204341166515)
    SquaredExponential(0.4605408503806978)
    Periodic(0.36971206610858265, 0.043922286046111925)
    Constant(0.29677976604794565)
    Times(Linear(0.8414273589795225), Plus(SquaredExponential(0.3394903984520361), SquaredExponential(0.7374494077480114), 3), 5)
    Times(Plus(Linear(0.2879310490732243), Linear(0.14471259775478984), 3), Constant(0.4829147415452433), 5)
    Plus(Linear(0.5199558938036932), SquaredExponential(0.9319057133205457), 3)
    Times(Linear(0.07266631444435401), Periodic(0.15731503797328927, 0.9270589187246194), 3)


We see that there's a bias towards shorter programs. (Do you see why this is?) But we also see that longer programs are possible, including programs that include multiple `Times` or `Plus` nodes.

### Adding a likelihood to obtain a complete model

We now connect the kernel program to the observed data, by using it to compute
a covariance matrix and sampling from a multivariate normal distribution.


```julia
@dist gamma_bounded_below(shape, scale, bound) = gamma(shape, scale) + bound

# Full model
@gen function model(xs::Vector{Float64})

    # Generate a covariance kernel
    covariance_fn = {:tree} ~ covariance_prior()

    # Sample a global noise level
    noise ~ gamma_bounded_below(1, 1, 0.01)

    # Compute the covariance between every pair (xs[i], xs[j])
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, xs)

    # Sample from the GP using a multivariate normal distribution with
    # the kernel-derived covariance matrix.
    ys ~ mvnormal(zeros(length(xs)), cov_matrix)

    # Return the covariance function, for easy printing.
    return covariance_fn
end;
```

Let's visualize a collection of traces from the model:


```julia
function serialize_trace(tr, xmin, xmax)
    (xs,) = get_args(tr)
    curveXs = collect(Float64, range(xmin, length=100, stop=xmax))
    curveYs = [predict_ys(get_retval(tr), 0.00001, xs, tr[:ys],curveXs) for i=1:5]
    Dict("y-coords" => tr[:ys], "curveXs" => curveXs, "curveYs" => curveYs)
end
```




    serialize_trace (generic function with 1 method)




```julia
viz = Viz(server, "gp-viz/dist", [collect(Float64, -1:0.1:1)]);
for iter=1:20
    (tr, _) = generate(model, (collect(Float64, -1:0.1:1),))
    putTrace!(viz, "t$(iter)", serialize_trace(tr, -1, 1))
end
displayInNotebook(viz)
```


body, html { margin-left: 0px; margin-right: 0px; }
h1[data-v-678d3a0d] { font-family: Avenir, Helvetica, Arial, sans-serif; text-align: center; }
#traces[data-v-678d3a0d] { display: flex; -moz-box-orient: horizontal; -moz-box-direction: normal; flex-flow: row wrap; }
#app { font-family: Avenir, Helvetica, Arial, sans-serif; color: rgb(44, 62, 80); }</style>


As we can see, many different patterns can be expressed in the kernel language we developed.
Depending on the noise level, these patterns (green) may be more or less evident in the
generated dataset (gray).

## 5. A tree regeneration proposal <a name="tree-regen" />

We will use MCMC to find good covariance functions given a dataset.

First, we write a helper to generate an initial trace for a set of `xs`
and `ys`:


```julia
function initialize_trace(xs::Vector{Float64}, ys::Vector{Float64})
    tr, = generate(model, (xs,), choicemap(:ys => ys))
    return tr
end
```




    initialize_trace (generic function with 1 method)



Next, we write a helper generative function that randomly chooses a node
in a kernel program's syntax tree. It starts at the root, and flips a coin
(at the address `:stop`) to decide whether to stop at the current node,
or to descend further down the tree. (If it reaches a leaf node, it's done,
and `:stop` is `true` with 100% probability. Otherwise, the coin is fair.)

The generative function _returns_ a hierarchical address specifying where the
chosen node was sampled in the trace of the model. (This is for convenience later on.)


```julia
@gen function random_node_path(n::Kernel)
    if ({:stop} ~ bernoulli(isa(n, PrimitiveKernel) ? 1.0 : 0.5))
        return :tree
    else
        (next_node, direction) = ({:left} ~ bernoulli(0.5)) ? (n.left, :left) : (n.right, :right)
        rest_of_path ~ random_node_path(next_node)

        if isa(rest_of_path, Pair)
            return :tree => direction => rest_of_path[2]
        else
            return :tree => direction
        end

    end
end;
```

To understand how this works, let's generate a sample model trace.
(To make things interesting, we'll generate one where the first
kernel node is guaranteed to be `Plus`.)


```julia
example_model_trace, = generate(model, ([0.],), choicemap((:tree => :kernel_type) => Plus));
println(get_choices(example_model_trace))
```

    │
    ├── :ys : [0.30854842303657554]
    │
    ├── :noise : 0.25291837958899466
    │
    └── :tree
        │
        ├── :kernel_type : Plus
        │
        ├── :left
        │   │
        │   ├── :kernel_type : Times
        │   │
        │   ├── :left
        │   │   │
        │   │   ├── :kernel_type : Plus
        │   │   │
        │   │   ├── :left
        │   │   │   │
        │   │   │   ├── :period : 0.1507566986392912
        │   │   │   │
        │   │   │   ├── :scale : 0.28140403604061714
        │   │   │   │
        │   │   │   └── :kernel_type : Periodic
        │   │   │
        │   │   └── :right
        │   │       │
        │   │       ├── :kernel_type : Plus
        │   │       │
        │   │       ├── :left
        │   │       │   │
        │   │       │   ├── :kernel_type : Plus
        │   │       │   │
        │   │       │   ├── :left
        │   │       │   │   │
        │   │       │   │   ├── :period : 0.6386290833062693
        │   │       │   │   │
        │   │       │   │   ├── :scale : 0.995743922941488
        │   │       │   │   │
        │   │       │   │   └── :kernel_type : Periodic
        │   │       │   │
        │   │       │   └── :right
        │   │       │       │
        │   │       │       ├── :kernel_type : Plus
        │   │       │       │
        │   │       │       ├── :left
        │   │       │       │   │
        │   │       │       │   ├── :param : 0.12094289809599923
        │   │       │       │   │
        │   │       │       │   └── :kernel_type : Constant
        │   │       │       │
        │   │       │       └── :right
        │   │       │           │
        │   │       │           ├── :period : 0.9878766934948584
        │   │       │           │
        │   │       │           ├── :scale : 0.10572873353479673
        │   │       │           │
        │   │       │           └── :kernel_type : Periodic
        │   │       │
        │   │       └── :right
        │   │           │
        │   │           ├── :param : 0.17316812396094905
        │   │           │
        │   │           └── :kernel_type : Constant
        │   │
        │   └── :right
        │       │
        │       ├── :period : 0.2292549565782267
        │       │
        │       ├── :scale : 0.29754214181188243
        │       │
        │       └── :kernel_type : Periodic
        │
        └── :right
            │
            ├── :param : 0.12854487730712116
            │
            └── :kernel_type : Linear



The returned covariance kernel program is:


```julia
example_covariance_kernel = get_retval(example_model_trace)
```




    Plus(Times(Plus(Periodic(0.28140403604061714, 0.1507566986392912), Plus(Plus(Periodic(0.995743922941488, 0.6386290833062693), Plus(Constant(0.12094289809599923), Periodic(0.10572873353479673, 0.9878766934948584), 3), 5), Constant(0.17316812396094905), 7), 9), Periodic(0.29754214181188243, 0.2292549565782267), 11), Linear(0.12854487730712116), 13)



Now, we can run our random node chooser:


```julia
random_node_chooser_trace = simulate(random_node_path, (example_covariance_kernel,));
```

Its return value is a path to a node in our model:


```julia
get_retval(random_node_chooser_trace)
```




    :tree => :left



Its trace shows _how_ it arrived at this path.


```julia
println(get_choices(random_node_chooser_trace))
```

    │
    ├── :left : true
    │
    ├── :stop : false
    │
    └── :rest_of_path
        │
        └── :stop : true



Feel free to run the cells above a few times to get a hang for what's going on.

### An involution proposal for tree regeneration

We will use `random_node_path` to implement a _random subtree regeneration_ proposal.

It will:
* Choose a node using `random_node_path`.
* Generate a brand new tree, using `covariance_prior()`, to replace the subtree at the chosen node.

We begin by writing the "proposal randomness" function:


```julia
@gen function regen_random_subtree_randomness(prev_trace)
    path ~ random_node_path(get_retval(prev_trace))
    new_subtree ~ covariance_prior()
    return path
end;
```

And now we write the involution. Recall that the third argument
that the involution accepts is the _proposal return value_, in our
case, a model address (like `:tree => :left`) corresponding to where
a particular node and its children (i.e., a subtree) were generated
in the model trace.


```julia
function subtree_involution(trace, forward_choices, path_to_subtree, proposal_args)
    # Need to return a new trace, backward_choices, and a weight.
    backward_choices = choicemap()

    # In the backward direction, the `random_node_path` function should
    # make all the same choices, so that the same exact node is reached
    # for resimulation.
    set_submap!(backward_choices, :path, get_submap(forward_choices, :path))

    # But in the backward direction, the `:new_subtree` generation should
    # produce the *existing* subtree.
    set_submap!(backward_choices, :new_subtree, get_submap(get_choices(trace), path_to_subtree))

    # The new trace should be just like the old one, but we are updating everything
    # about the new subtree.
    new_trace_choices = choicemap()
    set_submap!(new_trace_choices, path_to_subtree, get_submap(forward_choices, :new_subtree))

    # Run update and get the new weight.
    new_trace, weight, = update(trace, get_args(trace), (UnknownChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
end
```




    subtree_involution (generic function with 1 method)



We'll run our MCMC on a dataset of airline passenger volume over a number of years.


```julia
#########################
# load airline data set #
#########################
import CSV
using StatsBase: mean
function get_airline_dataset()
    df = CSV.read("$(@__DIR__)/airline.csv")
    xs = collect(df[!, 1])
    ys = collect(df[!, 2])
    xs .-= minimum(xs) # set x minimum to 0.
    xs /= maximum(xs) # scale x so that maximum is at 1.
    ys .-= mean(ys) # set y mean to 0.
    ys *= 4 / (maximum(ys) - minimum(ys)) # make it fit in the window [-2, 2]
    return (xs, ys)
end
```

    ┌ Info: Precompiling CSV [336ed68f-0bac-5ca0-87d4-7b16caf5d00b]
    └ @ Base loading.jl:1273





    get_airline_dataset (generic function with 1 method)




```julia
(xs, ys) = get_airline_dataset();
```

Let's run it:


```julia
function run_mcmc(trace, iters::Int, viz)
    for iter=1:iters
        (trace, acc) = mh(trace, regen_random_subtree_randomness, (), subtree_involution)
        (trace, _) = mh(trace, select(:noise))
        if acc
            putTrace!(viz, "t", serialize_trace(trace, -1, 2))
            sleep(0.1)
        end
    end
    return trace
end
```




    run_mcmc (generic function with 1 method)




```julia
viz = Viz(server, "gp-viz/dist", [xs])
t = initialize_trace(xs, ys);
putTrace!(viz, "t", serialize_trace(t, -1, 2))
t = displayInNotebook(viz) do
    sleep(1)
    run_mcmc(t, 2000, viz)
end;
```



```julia
println("Score was: ", get_score(t))
println("Program:")
println(get_retval(t))
```

    Score was: -189.7176826223711
    Program:
    Linear(0.722467023162884)



```julia

```
