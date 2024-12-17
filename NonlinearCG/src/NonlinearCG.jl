module NonlinearCG

include("optimizer.pluto.jl")

export optimize, MOP, ArmijoBacktracking, SteepestDescentDirection
export FletcherReevesRestart, FletcherReevesFractionalLP1,
FletcherReevesFractionalLP2, FletcherReevesBalancingOffset
export PRP3, PRPConeProjection

end
