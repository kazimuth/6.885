# Gen Problem Set

The following problem sets aim to teach modeling and inference in Gen.

### Problem Set 1
Problem Set 1 introduces modeling and inference in Gen. It consists of two
notebooks, one on the basics of modeling with probabilistic programs, and
one on performing certain kinds of iterative inference (e.g., MCMC) with
Gen's programmable inference library:

1. [Problem Set 1A: Introduction to Modeling in Gen](pset-modeling-introduction/Introduction%20to%20Modeling%20in%20Gen.jl)

2. [Problem Set 1B: Iterative inference in Gen](pset-iterative-inference/Iterative%20inference%20in%20Gen.jl)

### Problem Set 2

The Problem Set 2 notebooks will not be available until Problem Set 2 is
released.

3. [Problem Set 2A: Particle Filtering in Gen](pset-smc/Particle%20Filtering%20in%20Gen.jl)

4. [Problem Set 2B: Modeling with TensorFlow Code](pset-modeling-with-foreign-code/Modeling%20with%20TensorFlow%20Code.jl)

5. [Problem Set 2C: Modeling with black-box Julia code](pset-modeling-with-foreign-code/Modeling%20with%20Black-Box%20Julia%20Code.jl)

6. [Problem Set 2D: Data-driven proposals in Gen](pset-data-driven/Data-Driven%20Proposals%20in%20Gen.jl)


### Jupytext and auto-generated `.ipynb` files

These problem sets and accompanying Docker image make use of
[Jupytext](https://github.com/mwouts/jupytext), which converts bidirectionally
between a Jupyter notebook (`.ipynb`) and a Julia script (`.jl`).  Every time
you save your notebook in Jupyter, _both_ a `.jl` file and an `.ipynb` file
will be saved.

Both the `.ipynb` file and the `.jl` file are editable in the Jupyter web
interface.  To avoid merge conflicts and data loss, we suggest you edit only
the `.jl` file.  (Or at least, don't edit both files simultaneously.  And don't
worry, Jupyter will show a warning box if it detects that either file has
changed under its feet.)

If you use version control (git commits) to stage your work, you will find
that the `.jl` file is convenient for viewing a readable diff of your changes.
When you run cells, the cell outputs (print-outs and plots) are not saved in
the `.jl`, they are only saved to the `.ipynb`.


### Automated grading

Most exercises will be automatically graded. The grading is done using Julia's
built-in testing libraries. Each problem will run a smoke test and the grading.
Smoke tests should ensure that the format of the answer is correct. For
example, it could test if the relevant random choices exists in a Gen function
that implements a solution.

The tests are not expected to pass before answers are filled in. Note that some
of the tests in the automatic grading program are stochastic and may fail with
a very low probability despite the correct answer. In this case, we will ask
students to run the auto-grading function again.


### Submitting your problem set

* Run every problem set notebook end to end -- including the calls to
  auto-grading at the bottom of each notebook. Then, download the notebook as
  .html file via the File menu (File -> Download as -> HTML (.html). Check the
  resulting html files to see if the output of each Jupyter cell is correct.

* Ensure that the cells were executed in order by checking the counter to left
  of every cell (i.e. the execution count)

* Do not add content that is not necessary for the solution of the problem.

* Comment your code.

* Create a tarball will all of those .html files and submit them
  [here](https://airtable.com/shrLVvny6W0cwY4nB).

* Do not submit additional files.


### References

- [Gen documentation](https://probcomp.github.io/Gen/dev/)

- [GenTF documentation](https://probcomp.github.io/GenTF/dev/)

- [GenViz documentation](https://github.com/probcomp/GenViz)
