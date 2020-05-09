using Pkg
pkg"activate ."

using Reduce
@force using Reduce.Algebra

##
Reduce.squash(:(if true 1 else 2 end))
##
Algebra.solve(:(x^2==1),:x)
