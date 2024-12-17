# This file is meant to be formated with Literate.jl #src
import UnPack: @unpack
import LinearAlgebra as LA


# # Custom Frank-Wolfe Solver...
# ... to compute the multi-objective steepest descent direction cheaply.
# For unconstrained problems, the direction can be computed by projecting 
# ``\symbf{0}\in ℝ^n`` onto the convex hull of the negative objective gradients.
# This can be done easily with `JuMP` and a suitable solver (e.g., `COSMO`).
#
# In “Multi-Task Learning as Multi-Objective Optimization” by Sener & Koltun, the authors
# employ a variant of the Frank-Wolfe-type algorithms defined in 
# “Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization” by Jaggi.

#=
The objective for the projection problem is 
```math
F(\symbf{α}) 
= \frac{1}{2} ‖ \sum_{i=1}^K αᵢ ∇fᵢ ‖_2^2
= \frac{1}{2} ‖ \nabla \symbf{f}^T \symbf{α} ‖_2^2
= \frac{1}{2} \symbf a^T \nabla \symbf{f} \nabla \symbf{f}^T \symbf {α}
```
Hence,
```math
\nabla F(\symbf{α}) 
= \nabla \symbf{f} \nabla \symbf{f}^T \symbf α 
=: \symbf M \symbf α
```
=#
#=
The algorithm starts with some initial ``\symbf{α} = [α_1, …, α_K]^T`` 
and optimizes ``F`` within the standard simplex 
``S = \{\symbf α = [α_1, …, α_k]: α_i \ge 0, \sum_i α_i = 1\}.``
This leads to the following procedure:

1) Compute seed ``s`` as the minimizer of 
   ``\langle \symbf s, \nabla F(\symbf α_k) \rangle = 
   \langle \symbf s, \symbf M \symbf α_k\rangle``
   in ``S``. 
   The minimum is attained in one of the corners, i.e.,
   ``\symbf s = \symbf e_t``, where ``t`` is the minimizing index for the entries of ``\symbf M \symbf α_k``.
2) Compute the exact stepsize ``γ\in[0,1]`` that minimizes
   ```math
   F((1-γ)\symbf α_k + γ \symbf s).
   ```
3) Set ``\symbf α_{k+1} = (1-γ_k) α_k + γ_k \symbf s``.
=#

#=
## Finding the Stepsize

Let's discuss step 2. 
Luckily, we can indeed (easily) compute the minimizing stepsize.
Suppose ``\symbf v ∈ ℝⁿ`` and ``\symbf u ∈ ℝⁿ`` are vectors and 
``\symbf M ∈ ℝ^{n×n}`` is a **symmetric** 
square matrix. What is the minimum of the following function?
```math
σ(γ) = ( (1-γ) \symbf v + γ \symbf u )ᵀ \symbf M ( (1-γ) \symbf v + γ \symbf u) \qquad  (γ ∈ [0,1])
```

We have
```math
σ(γ) \begin{aligned}[t]
	&=
	( (1-γ) \symbf v + γ \symbf u )ᵀ\symbf{M} ( (1-γ) \symbf v + γ \symbf{u})
		\\
	&=
	(1-γ)² \underbrace{\symbf{v}ᵀ\symbf{M} \symbf{v}}_{a} + 
	  2γ(1-γ) \underbrace{\symbf{u}ᵀ\symbf{M} \symbf{v}}_{b} + 
	    γ² \underbrace{\symbf{u}ᵀ\symbf{M} \symbf{u}}_{c}
		\\
	&=
	(1 + γ² - 2γ)a + (2γ - 2γ²)b + γ² c 
		\\
	&=
	(a -2b + c) γ² + 2 (b-a) γ + a
\end{aligned}
```
The variables $a, b$ and $c$ are scalar.
The boundary values are
```math
σ₀ = σ(0) = a \text{and} σ₁ = σ(1) = c.
```
If ``(a-2b+c) > 0 ⇔ a-b > b-c``, 
then the parabola is convex and has its global minimum where the derivative is zero:
```math
2(a - 2b + c) y^* + 2(b-a) \stackrel{!}= 0 
 ⇔ 
	γ^* = \frac{-2(b-a)}{2(a -2 b + c)} 
		= \frac{a-b}{(a-b)+(c-b)}
```
If ``a-b < b -c``, the parabola is concave and this is a maximum.
The extremal value is 
```math
σ_* = σ(γ^*) 
	= \frac{(a - b)^2}{(a-b)+(c-b)} - \frac{2(a-b)^2}{(a-b) + (c-b)} + a
	= a - \frac{(a-b)^2}{(a-b) + (c-b)}
```
=#

"""
	min_quad(a,b,c)

Given a quadratic function ``(a -2b + c) γ² + 2 (b-a) γ + a`` with ``γ ∈ [0,1]``, return 
`γ_opt` minimizing the function in that interval and its optimal value `σ_opt`.
"""
function min_quad(a,b,c)
	a_min_b = a-b
	b_min_c = b-c
	if a_min_b > b_min_c
		## the function is a convex parabola and has its global minimum at `γ`
		γ = a_min_b /(a_min_b - b_min_c)
		if 0 < γ < 1
			## if its in the interval, return it
			σ = a - a_min_b * γ
			return γ, σ
		end
	end
	## the function is either a line or a concave parabola, the minimum is attained at the 
	## boundaries
	if a <= c
		return 0, a
	else
		return 1, c
	end
end

# To use the above function in the Frank-Wolfe algorithm, we define a 
# helper according to the definitions of ``a``, ``b`` and ``c``:
function min_chull2(M, v, u)
	Mv = M*v
	a = v'Mv
	b = u'Mv
	c = u'M*u
	return min_quad(a,b,c)
end

# ## Completed Algorithm

# The stepsize computation is the most difficult part. 
# Now, we only have to care about stopping and can complete the solver 
# for our sub-problem.

# First, we want to collect and pre-cache all arrays:
struct FrankWolfeCache{T<:AbstractFloat}
	α :: Vector{T}	# solution vector
	_α :: Vector{T}	# temporary vector
	M :: Matrix{T}	# symmetric matrix for gradient products
	#u :: Vector{T}	# seed vector
end

function init_frank_wolfe_cache(::Type{T}, num_grads) where T<:AbstractFloat
	α = zeros(num_grads)
	_α = similar(α)
	M = zeros(num_grads, num_grads)
	#u = similar(α)
	return FrankWolfeCache(α, _α, M)
end

# We initialize the cache from the Jacobian:
function frank_wolfe_multidir_dual(jac::AbstractMatrix{F}; kwargs...) where F<:Real
	T = Base.promote_type(Float32, F)
	num_grads = size(jac, 1)
	fw_cache = init_frank_wolfe_cache(T, num_grads)
	frank_wolfe_multidir_dual!(fw_cache, jac; kwargs...)
	return fw_cache.α
end

# Then we forward to the main algo:
function frank_wolfe_multidir_dual!(
	fw_cache::FrankWolfeCache, jac::AbstractMatrix; kwargs...
)
	@unpack α, _α, M = fw_cache
	return frank_wolfe_multidir_dual!(
		α, _α, M, jac; kwargs...
	)
end

# The main algo looks like this:
function frank_wolfe_multidir_dual!(
	α :: AbstractVector,	# solution vector
	_α :: AbstractVector,	# temporary vector
	M :: AbstractMatrix,	# symmetric product matrix for gradients
	## u :: AbstractVector,
	jac :: AbstractMatrix; 
	max_iter=50_000,
	eps_rel= let et=eltype(α); et <: AbstractFloat ? sqrt(eps(et)) : sqrt(eps(Float64)) end,
	eps_abs=0,
	ensure_descent=true,
)
	num_grads, num_vars = size(jac)

	## 1) Initialize ``α`` vector. There are smarter ways to do this...
	α .= 1/num_grads

	## 2) Build symmetric matrix of gradient-gradient products
	for (i,gi) = enumerate(eachrow(jac))
		for (j, gj) = enumerate(eachrow(jac))
			j<i && continue
			M[i, j] = M[j, i] = gi'gj 
		end
	end

	## 3) Solver iteration
	do_break = false
	_α .= α
	LA.mul!(α, M, _α)
	tval, t = findmin( α )

	it = 0
	for outer it=1:max_iter
		#=
		_α .= α
		LA.mul!(α, M, _α)
		tval, t = findmin( α )
		=#

		a = _α'α
		b = tval
		c = M[t, t]
		γ, _ = min_quad(a,b,c)

		#γ, _ = min_chull2(M, v, u)

		α .= _α
		α .*= (1-γ)
		α[t] += γ

		abs_change = sum( abs.( _α .- α ) )
		abs_rhs = sum( abs.(_α) )

		_α .= α
		LA.mul!(α, M, _α)
		tval, t = findmin( α )

		if abs_change <= eps_abs || abs_change <= eps_rel * abs_rhs
			if ensure_descent
				do_break = (tval >= 0)
			else
				do_break = true
			end
		end

		if do_break
			α .= _α
			break
		end

	end
	if !do_break
		α .= _α
	end
	α *= -1

	return α
end
