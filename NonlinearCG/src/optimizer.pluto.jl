### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 01754ba7-edda-4304-b98a-040ded8231dc
# ╠═╡ skip_as_script = true
#=╠═╡
using WGLMakie
  ╠═╡ =#

# ╔═╡ e862a437-6ecf-46f8-adce-916b0215b0e2
include("multidir_frank_wolfe.jl")

# ╔═╡ 1d45f603-f9ca-4815-b94d-a7674c2fbb4b
# ╠═╡ skip_as_script = true
#=╠═╡
import PlutoUI: TableOfContents
  ╠═╡ =#

# ╔═╡ 737f0c57-6146-4677-98b0-251e412a7ad8
# ╠═╡ skip_as_script = true
#=╠═╡
TableOfContents()
  ╠═╡ =#

# ╔═╡ fea950e4-29c6-41ed-be89-a58b222f7973
import UnPack: @unpack

# ╔═╡ dfefc188-e6de-42fe-b782-7a4011fc1073
import Logging: LogLevel, Info, Debug, @logmsg

# ╔═╡ 4606c253-f824-4468-8c08-72ab120b720d
md"# Problem Type"

# ╔═╡ c6beceae-b713-11ef-0890-67d740e56c6c
md"""
We want to minimize multiple nonlinear, smooth objectives.
The unconstrained problem is 
```math
    \min_{x ∈ ℝ^N} 
        \begin{bmatrix}
			f_1(x)
			\\
            ⋮
            \\
            f_K(x)
        \end{bmatrix}
    \tag{MOP}
```
To perform optimization, we need a way to query metadata for (MOP), 
and evaluate and differentiate the objective functions.
"""

# ╔═╡ 07418323-ffe7-4053-8498-7a7bd9693917
md"## Implementation"

# ╔═╡ 423e5a3a-8869-4f8e-b42c-3e96f405aed8
md"### Immutable Problem Type"

# ╔═╡ 1c8552a3-2ace-4519-8287-5cf43b23f68b
md"""
The mathematical description $(MOP)$ is implemented by type `MOP`.
Julia functions don't have dimension metadata, so the user has to specify number of input variables `dim_in` and dimension of objective vector `dim_out`.
The objective function and the Jacobian function should be in-place.
"""

# ╔═╡ 2c24581f-af6f-4209-af8b-448f32c2fea7
Base.@kwdef struct MOP{
	objectives!_Type, 	# <: Function,
	jac!_Type, 			# <: Function,
	objectives_and_jac!_Type # <: Union{Function, Nothing}
}
	dim_in :: Int = 0
	dim_out :: Int = 0

	"In-place objective function with signature `objectives!(y::Vector{Float64}, x::Vector{Float64})`"
	objectives! :: objectives!_Type = nothing

	"In-place Jacobian function signature `jac!(Dy, x)`."
	jac! :: jac!_Type = nothing

	"Optional in-place objectives and jacobian evaluation function with arguments (y, Dy, x)"
	objectives_and_jac! :: objectives_and_jac!_Type = nothing
end

# ╔═╡ b56c13ea-0ea3-4865-9e1e-526cf07b8081
md"#### Metadata"

# ╔═╡ a1c06c57-6c3a-4ba9-85f5-ec1f1c4ef6e9
float_type(::MOP)=Float64

# ╔═╡ 8ba05829-a74a-4fd4-9310-87e9f147519a
md"Query dimension information:"

# ╔═╡ 48534191-f29a-45b4-a4c9-2e8854deb839
dim_in(mop::MOP)=mop.dim_in

# ╔═╡ 189d0a44-6671-4521-8d7e-c5584a6f8375
dim_out(mop::MOP)=mop.dim_out

# ╔═╡ d22d376b-e8b3-4a3c-9afa-3b43575bc1ea
md"#### Wrapped Problem Type"

# ╔═╡ f1dd1c2d-d521-4f2d-90bd-952379f3427d
md"We don't want to modify anything provided by the user.
At the same time, we'd like to count function evaluations.
We thus wrap the user provided problem."

# ╔═╡ cd6e18ca-f659-4100-a47d-409262660384
md"Forward metadata methods:"

# ╔═╡ cc255919-7bcd-48c3-8e3b-c86ec365c977
md"Query or set number of function calls:"

# ╔═╡ dd82f281-a10b-44ba-821a-a56f2923ae8b
md"### Evaluation"

# ╔═╡ a9b50868-2880-47df-904e-aed3891f73d4
md"# Algorithm"

# ╔═╡ 03d32a31-9a60-4556-a2e6-b048ab1160bc
md"## Types and Data"

# ╔═╡ 6773003d-51e6-43b8-bd05-06089bda2872
md"### Stop Codes"

# ╔═╡ 1a7c4eb1-2981-4a01-8526-37fb9a4ea05a
@enum STOP_CODE :: Int8 begin
    STOP_MAX_ITER
    STOP_BUDGET_FUNCS
    STOP_BUDGET_GRADS
    STOP_CRIT_TOL_ABS
    STOP_X_TOL_REL
    STOP_X_TOL_ABS
    STOP_FX_TOL_REL
    STOP_FX_TOL_ABS
end

# ╔═╡ 6db3895b-dae5-4d5d-a539-2667f932cf48
Base.@kwdef struct CountedMOP{
	mop_wrapped_Type
}
	mop_wrapped :: mop_wrapped_Type
	
	num_calls_objectives_ref :: Base.RefValue{Int} = Ref(0)
	num_calls_jac_ref :: Base.RefValue{Int} = Ref(0)

	# these are overwritten by optimization routine
	max_num_calls_objectives_ref :: Base.RefValue{Int} = Ref(typemax(Int))
	max_num_calls_jac_ref :: Base.RefValue{Int} = Ref(typemax(Int))
end

# ╔═╡ 3b38f68b-0a07-4898-a7b3-25affa633ed8
function wrap_mop(mop::MOP)
	return CountedMOP(; mop_wrapped = mop)
end

# ╔═╡ 955b2796-8279-4831-b12b-8fa091721dd5
float_type(mop::CountedMOP) = float_type(mop.mop_wrapped)

# ╔═╡ e29f3212-71ce-43f9-96ed-bc1597f2a88c
dim_in(mop::CountedMOP) = dim_in(mop.mop_wrapped)

# ╔═╡ 35ff7b4a-9bc8-4e26-8371-736eb04e4fba
function mop_assert_x(mop, x)
	@assert length(x) == dim_in(mop) "Length of input vector does not match `mop.dim_in`."
	nothing
end

# ╔═╡ 5d741ef6-b695-4915-8da6-1a50d943e875
dim_out(mop::CountedMOP) = dim_out(mop.mop_wrapped)

# ╔═╡ 9c09fc70-c270-4515-a2eb-241650f2c651
function mop_assert_y(mop, y)
	@assert length(y) == dim_out(mop) "Length of output vector does not match `mop.dim_out`."
	nothing
end

# ╔═╡ 9cf361e5-57f3-48e5-9405-a2c6f5df8b10
function mop_assert_Dy(mop, Dy)
	sz_Dy = size(Dy)
	sz_mop = (dim_out(mop), dim_in(mop))
	@assert sz_Dy == sz_mop "Size of Jacobian should be $(sz_mop), but is $(sz_Dy)."
	nothing
end

# ╔═╡ 0032a05a-3989-4ea0-ba2c-de683f0cbe4b
num_calls_objectives(mop::CountedMOP) = mop.num_calls_objectives_ref[]

# ╔═╡ 0277713e-bb2f-43d9-837e-583c944324fa
max_num_calls_objectives(mop::CountedMOP) = mop.max_num_calls_objectives_ref[]

# ╔═╡ 3ca6b8c2-94b2-4c71-890b-421b987bbf11
function inc_num_calls_objectives!(mop::CountedMOP)
	mop.num_calls_objectives_ref[] += 1
end

# ╔═╡ 8bcb7058-b529-4009-9874-db494183b7d2
function inc_num_calls_jac!(mop::CountedMOP)
	mop.num_calls_jac_ref[] += 1
end

# ╔═╡ c9ad3d97-330d-4fd7-a468-656f5cb578a5
num_calls_jac(mop::CountedMOP) = mop.num_calls_jac_ref[]

# ╔═╡ b972d504-3518-4b6e-8f0e-afd595f62a32
max_num_calls_jac(mop::CountedMOP) = mop.max_num_calls_jac_ref[]

# ╔═╡ 69ce9e0f-d336-4c55-a7d4-86ec941924cd
max_num_calls_objectives!(mop::CountedMOP, val) = (mop.max_num_calls_objectives_ref[] = val)

# ╔═╡ d9fee57d-ee9f-4ed7-aa85-9e61472ff4cd
max_num_calls_jac!(mop::CountedMOP, val) = (mop.max_num_calls_jac_ref[] = val)

# ╔═╡ 05cd3f07-2e41-4ebf-9cdf-bc32248c0aa5
function check_budget_objectives(mop)
	n = num_calls_objectives(mop)
	N = max_num_calls_objectives(mop)
	if n >= N
		return STOP_BUDGET_FUNCS
	end
	return nothing
end

# ╔═╡ 370b2e94-8628-4582-9a87-7d7e4bfa32ab
function objectives!(y, mop::CountedMOP, x)
	mop_assert_x(mop, x)
	mop_assert_y(mop, y)
	stop_code = check_budget_objectives(mop)
	!isnothing(stop_code) && return stop_code
	inc_num_calls_objectives!(mop)
	return mop.mop_wrapped.objectives!(y, x)
end

# ╔═╡ 52c01c37-130f-44d3-b9e2-b15af18cc251
function check_budget_jac(mop)
	n = num_calls_jac(mop)
	N = max_num_calls_jac(mop)
	if n >= N
		return STOP_BUDGET_GRADS
	end
	return nothing
end

# ╔═╡ 611df4e2-3c46-4ded-a222-9d3cbc46f1b7
function jac!(Dy, mop::CountedMOP, x)
	mop_assert_x(mop, x)
	mop_assert_Dy(mop, Dy)
	stop_code = check_budget_jac(mop)
	!isnothing(stop_code) && return stop_code
	inc_num_calls_jac!(mop)
	return mop.mop_wrapped.jac!(Dy, x)
end

# ╔═╡ 2f3fe957-9f8b-47cd-9f3c-6db914956fa1
function _objectives_and_jac!(y, Dy, mop, ::Nothing, x)
	objectives!(y, mop, x)
	jac!(Dy, mop, x)
	return nothing
end

# ╔═╡ 7e3406d3-8252-4b4e-bd9a-b1edb11a0d21
function _objectives_and_jac!(y, Dy, mop, mop_objectives_and_jac!, x)
	mop_assert_x(mop, x)
	mop_assert_y(mop, y)
	mop_assert_Dy(mop, Dy)
	stop_code = check_budget_objectives(mop)
	!isnothing(stop_code) && return stop_code
	stop_code = check_budget_jac(mop)
	!isnothing(stop_code) && return stop_code
	inc_num_calls_objectives!(mop)
	inc_num_calls_jac!(mop)
	return mop_objectives_and_jac!(y, Dy, x)
end

# ╔═╡ a58d490c-2164-410d-ac5a-241e4b8130ba
function objectives_and_jac!(y, Dy, mop::CountedMOP, x)
	return _objectives_and_jac!(y, Dy, mop, mop.mop_wrapped.objectives_and_jac!, x)
end

# ╔═╡ ad0b0559-41eb-49a1-a6ab-455201160975
md"""
Because we check stopping criteria quite often, there is the helper macro 
`@ignorebreak`.
"""

# ╔═╡ 14b9e1a0-435c-41af-9368-41ea8a899c97
## helper for `@ignorebreak`
function _parse_ignoraise_expr(ex)
	has_lhs = false
	if Meta.isexpr(ex, :(=), 2)
		lhs, rhs = esc.(ex.args)
		has_lhs = true
	else
		lhs = nothing	# not really necessary
		rhs = esc(ex)
	end
	return has_lhs, lhs, rhs
end

# ╔═╡ a00b4a6a-b93d-4859-abfb-be43c4d3558c
"""
    @ignorebreak do_something(args...)
    @ignorebreak lhs = do_something(args...)

If the expression `do_something` returns a `STOP_CODE`, then `break`.
Otherwise assign the result to `lhs`, if there is a left-hand side.
"""
macro ignorebreak(ex, indent_ex=0)
	has_lhs, lhs, rhs = _parse_ignoraise_expr(ex)
	return quote
		ret_val = $(rhs)
		do_break = ret_val isa STOP_CODE
		$(if has_lhs
			:($(lhs) = ret_val)
		else
			:(ret_val = nothing)
		end)
		do_break && break		
	end
end

# ╔═╡ ffb6dc4d-4f02-4127-ab01-f858ac12b805
md"### Common Arrays"

# ╔═╡ 06bb50f9-18b5-402b-ae2b-daae9ff96b83
md"We have a bunch of arrays that we need in many places:"

# ╔═╡ 2ff5e653-86b8-473b-9e35-b4091a298e89
struct CommonArrays{F<:AbstractFloat}
	"Variable vector."
	x :: Vector{F}
	"Objective vector."
	fx :: Vector{F}
	"Objective Jacobian matrix."
	Dfx :: Matrix{F}

	"Descent direction"
	d :: Vector{F}
	"Next variable vector"
	xd :: Vector{F}
	"Next objective vector"
	fxd :: Vector{F}
	"Displacement vector in objective space"
	y :: Vector{F}
end	

# ╔═╡ 38103ab8-b876-4ecf-8830-41249dd41904
md"The common arrays `carrays` come from this function:"

# ╔═╡ e557a091-18ab-4502-a419-c1cd14bc876b
function initialize_common_arrays(mop)
    F = float_type(mop)
    n_vars = dim_in(mop)
    n_objfs = dim_out(mop)

    x = zeros(F, n_vars)
    fx = zeros(F, n_objfs)
    Dfx = zeros(F, n_objfs, n_vars)

    d = similar(x)
    xd = similar(x)
    fxd = similar(fx)
	y = similar(fx)

    return CommonArrays(x, fx, Dfx, d, xd, fxd, y)
end

# ╔═╡ 36262f76-15de-4920-b5d3-bea05db849bd
md"### Steps"

# ╔═╡ 7fc47384-31b8-4fcf-b7f1-2ca9d8d1393c
md"""
Our algorithm is designed to be very flexible with regard to the actual descent steps used.
We take some configuration of type `AbstractStepRule`.
Then, at initialization a cache of type `AbstractStepRuleCache` is constructed.
We use the cache for storing temporary data and for dispatch, e.g., with `step!`
"""

# ╔═╡ 460b2c61-6e9c-43a2-aae6-92f1171c8660
abstract type AbstractStepRule end

# ╔═╡ 120acaa5-b79f-4914-9c8c-c47a0d844c83
abstract type AbstractStepRuleCache end

# ╔═╡ efbb21f2-a445-43c7-b5ea-213eed77c3da
md"Initialization function for cache corresponding to `step_rule`:"

# ╔═╡ 651abe09-9eeb-4f90-851e-23a0e32a6c96
function init_cache(step_rule::AbstractStepRule, mop)::AbstractStepRuleCache
    return nothing
end

# ╔═╡ 8f4507d2-1d12-42f5-953b-b8b5da8e30a5
md"""
Step computation function:
The object `carrays` are the **common arrays**.
The methods for `step!` must
* set `carrays.d` to contain the step vector in variable space, 
* `carrays.xd` to the next iteration vector, 
* and `carrays.fxd` to the next value vector.
"""

# ╔═╡ c4ca79fd-00b0-4d8c-8699-73b4dbc54f41
function step!(carrays, ::AbstractStepRuleCache, mop, it_index; kwargs...)
    nothing
end

# ╔═╡ 4e515bed-d72f-4865-a334-4b0c16768893
md"A function to return a criticality value, for stopping. It is called after `step!` as been executed."

# ╔═╡ 20872fb5-f12c-437b-821f-62b394987f07
criticality(carrays, ::AbstractStepRuleCache)=Inf

# ╔═╡ 2ccf8082-6427-4d13-b155-f340c1025d9f
md"### Callback"

# ╔═╡ 70e74fd5-2242-4fc7-9dae-59e48f1c77e0
md"""
Before giving the complete loop, we define a default (no-op) callback.
A user can provide their own callback function instead."""

# ╔═╡ 86ab5eb1-9c50-48fa-8fd3-575cc2e50eec
abstract type AbstractCallback end

# ╔═╡ 8d8caa14-d97c-4658-8e98-edbbcac2dca6
struct DefaultCallback <: AbstractCallback end

# ╔═╡ ac6d8197-fa82-43d4-9f06-0422d73e62c5
md"Before starting optimization, we allow for initialization:"

# ╔═╡ a837dc91-6a85-4a7a-8cdc-70f8b9b1d869
function initialize_callback(
    uninitialized_callback::AbstractCallback, carrays, mop, step_cache)::AbstractCallback
    return uninitialized_callback
end

# ╔═╡ d20e013f-f1c3-46b3-8cf6-4e07d202ae84
md"""
The function `exec_callback` is called at the end of each iteration.
If it returns a `STOP_CODE`, then we interrupt the algorithm and return.
"""

# ╔═╡ c7c62a50-4d7c-41b9-9905-2a5f6da55fab
function exec_callback(::AbstractCallback, it_index, carrays, mop, step_cache, stop_code)
    return nothing
end

# ╔═╡ 9a0646d2-1f30-4db7-9d3f-2ba3b7cf10a1
md"## Utility Functions"

# ╔═╡ a26663d2-730b-46f0-bedb-cc692e68516c
md"Prepare strings for logging with a nice helper function..."

# ╔═╡ 95e2caeb-fe58-424e-ac3a-7a01c86d36fe
import Printf: @sprintf

# ╔═╡ c0116a05-eef2-41a7-aed9-318188cb694a
function pretty_row_vec(
	x::AbstractVector;
	cutoff=80
)
	repr_str = "["
	lenx = length(x)
	for (i, xi) in enumerate(x)
		xi_str = @sprintf("%.2e", xi)
		if length(repr_str) + length(xi_str) >= cutoff
			repr_str *= "..."
			break
		end
		repr_str *= xi_str
		if i < lenx
			repr_str *= ", "
		end
	end
	repr_str *= "]"
	return repr_str
end

# ╔═╡ 5d8267cc-27d0-438f-b326-8a8583b54f4b
# fallback for arbitrary objects
pretty_row_vec(x; kwargs...)=string(x)

# ╔═╡ c813dbbd-e7c3-4308-9a15-5055bf9b14a7
md"## Main Loop"

# ╔═╡ beb2c714-4037-45d7-875a-34e89bf6deab
md"# Steps"

# ╔═╡ 192cedc1-f52d-49c4-aca3-29b2920e5a75
md"## Stepsizes"

# ╔═╡ 2cd698d7-dc83-4095-ac65-e63b563e965d
md"Within the `step!` implementation, we might wish to employ different stepsize methods.
We provide an interface similar to the `AbstractStepRule` interface."

# ╔═╡ 1fbad73b-046d-417f-b7ef-22287fb7b5d9
abstract type AbstractStepsizeRule end

# ╔═╡ c432d8e6-6148-44e1-bf29-73f172d2f1ec
abstract type AbstractStepsizeCache end

# ╔═╡ d6d001e3-6641-45c5-b295-1246d3397336
md"Initalization function for `AbstractStepsizeCache`:"

# ╔═╡ df3d1851-9a1f-489e-9f1e-b3baaa24a914
function init_cache(sz_rule::AbstractStepsizeRule, mop)::AbstractStepsizeCache
    nothing
end

# ╔═╡ c0642afa-96f1-49d4-8cae-16d34c5b62e3
md"Mutating stepsize function.
A `stepsize!` method must correctly set `d`, `xd` and `fxd`."

# ╔═╡ 4ae8c923-8af7-4d71-9354-1be96b5d0cfc
function stepsize!(carrays, ::AbstractStepsizeCache, mop; kwargs...)::Nothing
    error("`stepsize` not implemented.")
end

# ╔═╡ a406b503-262d-4773-960e-4d13fce5ff0d
md"### Fixed Stepsize"

# ╔═╡ 082a8d61-ac69-4ed7-addc-f8e951f80fed
Base.@kwdef struct FixedStepsize{F<:Real} <: AbstractStepsizeRule
    sz :: F = 0.001
end

# ╔═╡ cf720d41-d841-4ecd-a2f7-15d53b22a35d
struct FixedStepsizeCache{F<:AbstractFloat} <: AbstractStepsizeCache
    sz :: F
end

# ╔═╡ a0893f2c-1342-4099-a974-f883808f45df
function init_cache(sz_rule::FixedStepsize, mop)
    return FixedStepsizeCache(convert(float_type(mop), sz_rule.sz))
end

# ╔═╡ 16bdabcc-b652-4d64-b350-da65316d973e
function stepsize!(carrays, sz_cache::FixedStepsizeCache, mop; kwargs...)
    @unpack sz = sz_cache
    @unpack d, x, xd, fxd = carrays
    d .*= sz
    xd .= x .+ d
    return objectives!(fxd, mop, xd)
end

# ╔═╡ cb7497f7-1b31-4549-a6f4-019a90443ed1
md"### Armijo Backtracking"

# ╔═╡ 3a361a85-8469-4677-b70f-da49019035e4
md"We can also do Armijo-like backtracking.
We implement this ourselves, because we also offer the modified variant.
It is enabled by setting `is_modified = Val(true)`.
"

# ╔═╡ f03796a7-3ff1-477c-a4a4-8a099d6b92ce
Base.@kwdef struct ArmijoBacktracking{F<:Real} <: AbstractStepsizeRule
	"Backtracking factor."
    factor :: F = 0.5
	"Right hand side constant in Armijo test."
    constant :: F = 1e-4
	
	"Initial test stepsize."
    sz0 :: F = 1.0
	"Whether to reduce all objectives, any objective, or the difference of maximum values."
    mode :: Union{Val{:all}, Val{:any}, Val{:max}} = Val(:all)
	"Whether to use modified right hand side in Armijo test."
    is_modified :: Union{Val{true}, Val{false}} = Val{false}()
end

# ╔═╡ 76794911-fecb-44b8-8459-d52939145e55
md"The cache has all the same fields, strongly typed, and a temporary helper array:"

# ╔═╡ 831a2337-9c28-4b93-8871-b80b3b5dee77
struct ArmijoBacktrackingCache{F<:AbstractFloat} <: AbstractStepsizeCache
    factor :: F
    constant :: F
    sz0 :: F
    mode :: Union{Val{:all}, Val{:any}, Val{:max}}
    is_modified :: Union{Val{true}, Val{false}}
    lhs_vec :: Vector{F}
end

# ╔═╡ 6beffea4-098b-4f2e-9efd-87335321027b
function init_cache(sz_rule::ArmijoBacktracking, mop)
    F = float_type(mop)
    @unpack factor, constant, sz0, mode, is_modified = sz_rule
    return ArmijoBacktrackingCache{F}(
        factor, constant, sz0, mode, is_modified, zeros(F, dim_out(mop))
    )
end

# ╔═╡ 4dba2ef2-1c12-4e54-9557-0c50a98c6d74
md"#### Backtracking Functions"

# ╔═╡ f27c17a3-dc89-4041-9555-940dbfdefcdc
function _armijo_rhs(is_modified::Val{false}, d_norm2_sq, sz, constant)
	return constant * d_norm2_sq * sz
end

# ╔═╡ f79a2daa-1c83-453c-9d8f-2933b9aee56b
function _armijo_rhs(is_modified::Val{true}, d_norm2_sq, sz, constant)
	return constant * d_norm2_sq * sz^2
end

# ╔═╡ 0a224393-3dad-415d-a009-4d29a5bcf0af
function _armijo_apply_factor_to_rhs(is_modified::Val{false}, rhs, factor)
    return rhs * factor
end

# ╔═╡ 46d1e46b-3d2a-4b3f-8851-5995816a54c7
function _armijo_apply_factor_to_rhs(is_modified::Val{true}, rhs, factor)
    ## rhs = constant * sz^2 * critval
    ## _sz = (sz * factor) ⇒ _sz^2 = sz^2 * factor^2
    ## ⇒ _rhs = constant * _sz^2 * critval = factor^2 * rhs
    return rhs * factor^2
end

# ╔═╡ 36cb9a2c-1df8-4982-9ee0-50a5a687d76a
_armijo_φ(::Val{:all}, fx)=fx

# ╔═╡ 1ba108a2-65cc-4bbe-a6bd-63fc1c7e0531
_armijo_φ(::Val{:any}, fx)=fx

# ╔═╡ f0abfe44-86d1-406c-aed8-a4d1bb6e89b0
_armijo_φ(::Val{:max}, fx)=maximum(fx)

# ╔═╡ 5c1be58e-98aa-4367-8a1d-3f96dc3ccbd9
function _armijo_test(::Val{:all}, lhs_vec, φx, fxd, rhs)
    return all( lhs_vec .>= rhs )
end

# ╔═╡ 35c34c50-ab63-4493-9a8a-24ac1952540f
function _armijo_test(::Val{:any}, lhs_vec, φx, fxd, rhs)
    return maximum(lhs_vec) >= rhs
end

# ╔═╡ 91afbddf-2aba-4428-be04-e06020ad694b
function _armijo_test(::Val{:max}, lhs_vec, φx, fxd, rhs)
    return φx - maximum(fxd) >= rhs
end

# ╔═╡ 92e74936-9522-480d-baf6-6ff1836e2c43
function armijo_backtrack!(
	## modified:
    d, xd, fxd,
	## not-modified
    mop,
    x, fx,
    factor, constant, sz0, mode, lhs_vec,
    is_modified :: Union{Val{true}, Val{false}} = Val{false}();
    x_tol_abs = 0,
    x_tol_rel = 0,
    kwargs...
)
	## prevent infinite loops by means of a finite tolerance
	if x_tol_abs <= 0
		x_tol_abs = mapreduce(eps, min, x)
	end

	## initialize loop vars
	d_norm2_sq = sum( d.^2 )
    zero_step = d_norm2_sq <= 0
    stop_code = nothing
    
    if !zero_step
		sz = sz0
        rhs = _armijo_rhs(is_modified, d_norm2_sq, sz, constant)

        d .*= sz
        d_norm = LA.norm(d, Inf)

    	xd .= x .+ d
        stop_code = objectives!(fxd, mop, xd)
        if stop_code isa STOP_CODE
            zero_step = true
        end
    end

    if !zero_step	
		## these are constant during while loop:
    	x_norm = LA.norm(x, Inf)
		min_d_norm = min(x_tol_abs, x_tol_rel * x_norm)
	    φx = _armijo_φ(mode, fx)
        while true
            if sz <= 0
                zero_step = true
                break
            end

            if d_norm <= min_d_norm
                break
            end
            
            lhs_vec .= fx .- fxd

			## φx - φxd ⪸ rhs
            if _armijo_test(mode, lhs_vec, φx, fxd, rhs)
                break
            end
            
            rhs = _armijo_apply_factor_to_rhs(is_modified, rhs, factor)
            sz *= factor
            d .*= factor
            d_norm *= factor
            xd .= x .+ d
            stop_code = objectives!(fxd, mop, xd)
            if stop_code isa STOP_CODE
                zero_step = true
                break
            end
        end
    end
    if zero_step
        sz = 0
        d .= 0
        xd .= x
        fxd .= fx
    end

    if !(stop_code isa STOP_CODE)
        stop_code = sz
    end
    return stop_code 
end

# ╔═╡ 14cf9e93-8769-4de5-a076-4e8237e03e1f
function stepsize!(
    carrays, sz_cache::ArmijoBacktrackingCache, mop;
    kwargs...
)
    @unpack d, xd, fxd, x, fx = carrays
    @unpack factor, constant, sz0, mode, is_modified, lhs_vec = sz_cache
    return armijo_backtrack!(
        d, xd, fxd, mop, x, fx, factor, constant, sz0, mode, lhs_vec, is_modified;
        kwargs...
    )
end

# ╔═╡ 0c49efa6-cd89-493e-8b1a-85571aae2761
md"## Directions"

# ╔═╡ d7ba86be-ad06-490e-bac8-908fd0638b29
md"### Steepest Descent"

# ╔═╡ 8bae3396-36e7-4532-a946-ebbbd9b5a5d4
md"""
```math
\mathbf δ = \operatorname*{argmin}_{\mathbf d \in ℝ^n } \mathbb D(\mathbf d) + \frac{1}{2}\| \mathbf d\|^2
```
"""

# ╔═╡ 492ff20d-64af-49ad-8a0f-9799fee41051
md"Import script to solve convex quadratic optimization problem:"

# ╔═╡ 49f21216-679d-4f39-b9e4-089ac5b7ebb0
md"Define type for dispatch and configuration:"

# ╔═╡ de7628ff-ce1d-451f-8eec-0e956a42e4eb
Base.@kwdef struct SteepestDescentDirection{
    sz_rule_Type, 
} <: AbstractStepRule
    sz_rule :: sz_rule_Type = ArmijoBacktracking(;
		is_modified = Val(false)
	)
end

# ╔═╡ 589768dd-331c-4da0-a11e-d0c2c0f40984
md"
Many other direction schemes also depend on the steepest descent direction data.
We define a common cache for all these directions."

# ╔═╡ 60e85226-cdfe-4e57-b489-124bc84c7120
struct CommonStepCache{
    F<:AbstractFloat,
    sz_cache_Type,
    fw_cache_Type,
}
	criticality_ref :: Base.RefValue{F}
    sz_cache :: sz_cache_Type
    fw_cache :: fw_cache_Type
end

# ╔═╡ bd8a0cb0-926a-4824-9a32-ff8127eeff61
md"The common step cache `ccache` has al the arrays for computing the steepest descent direction:"

# ╔═╡ a3571fc6-12ff-4827-8b99-7e2df8bcd187
function steepest_descent_direction!(carrays::CommonArrays, ccache::CommonStepCache)
    @unpack d, Dfx = carrays
    @unpack criticality_ref, fw_cache = ccache

    critval = criticality_ref[] = steepest_descent_direction!(d, fw_cache, Dfx)
    return critval
end

# ╔═╡ 893a3e63-72e1-46a3-bc3c-9eab65d347f5
function maxdot(Dfx, d)
    return mapreduce(Base.Fix1(LA.dot, d), max, eachrow(Dfx))
end

# ╔═╡ e4148017-12b9-4084-a88a-94c4f325ddf2
function steepest_descent_direction!(d, fw_cache, Dfx)
	## compute (negative) KKT multipliers for steepest descent direction
    α = frank_wolfe_multidir_dual!(fw_cache, Dfx)
    
    ## use these to set steepest descent direction `d`
    LA.mul!(d, Dfx', α)

    ## before scaling `d`, set criticality
    critval = abs(maxdot(Dfx, d))
	return critval
end

# ╔═╡ 9bb1d034-58f2-463b-aa7b-f0630ab52f5d
md"The actual cache for `SteepestDescentDirection` is basically just a wrapper:"

# ╔═╡ de5cff65-cf87-4ff6-bc98-56278c5e8539
struct SteepestDescentDirectionCache{
    ccache_Type <: CommonStepCache,
    step_meta_Type
} <: AbstractStepRuleCache
    ccache :: ccache_Type
    step_meta :: step_meta_Type 	# not used for now
end

# ╔═╡ 707a6e6e-d956-4261-8db1-83987cab8a1b
function criticality(carrays, step_cache::SteepestDescentDirectionCache)
    return step_cache.ccache.criticality_ref[]
end

# ╔═╡ 3fd9a9ef-b268-4b22-b868-e7858152ae26
function step!(
    carrays, step_cache::SteepestDescentDirectionCache, mop, it_index; 
    kwargs...
)
    @unpack ccache, step_meta = step_cache
    
    ## modify carrays.d to be steepest descent direction
	## set carrays.critval_ref[] to hold criticality
    critval_sd = steepest_descent_direction!(carrays, ccache)

    ## compute a stepsize, scale `d`, set `xd .= x .+ d` and values `fxd`
    @unpack sz_cache = ccache
    return stepsize!(carrays, sz_cache, mop; kwargs...)
end

# ╔═╡ e692c5ca-57d5-494a-9469-006109c0ca02
md"### Fletcher Reeves Restart"

# ╔═╡ 7723b535-3cc4-44b3-8b21-c0f0490b5ebf
Base.@kwdef struct FletcherReevesRestart{
    F<:Real,
    sz_ruleType
} <: AbstractStepRule
    sz_rule :: sz_ruleType = ArmijoBacktracking(; is_modified=Val{true}())
    critval_mode :: Union{Val{:sd}, Val{:cg}} = Val{:sd}()
    wolfe_constant :: F = .9
end

# ╔═╡ 91a999db-6428-43c1-975b-7859b0af6c00
abstract type AbstractCGCache <: AbstractStepRuleCache end

# ╔═╡ 9ea12970-eca2-47ff-8dee-90c222af5c9e
struct FletcherReevesRestartCache{
    F<:AbstractFloat,
    ccacheType
} <: AbstractCGCache
    ccache :: ccacheType
    critval_mode :: Union{Val{:sd}, Val{:cg}}
    criticality_ref :: Base.RefValue{F}
    wolfe_constant :: F
    d_prev :: Vector{F}
end

# ╔═╡ 92d896fe-7066-4fc7-a2fd-cf91ccc9a466
function _cg_criticality(step_cache)
    return _cg_criticality(step_cache.critval_mode, step_cache)
end

# ╔═╡ 37cfa713-1fc5-4878-baa1-2b31a348a505
function _cg_criticality(::Val{:sd}, step_cache)
    return step_cache.ccache.criticality_ref[]
end

# ╔═╡ 44d26cf8-9f54-4604-8a4d-c85a65ce2d5b
function _cg_criticality(::Val{:cg}, step_cache)
    return step_cache.criticality_ref[]
end

# ╔═╡ f5388f8d-4013-491d-9c24-58d5c19cf9a5
function criticality(carrays, step_cache::AbstractCGCache)
    return _cg_criticality(step_cache)
end

# ╔═╡ 091bf95f-b0e6-4ff6-b1ac-4b765eacffb3
function step!(
    carrays, step_cache::FletcherReevesRestartCache, mop, it_index; 
    kwargs...
)
    @unpack ccache = step_cache
    
    ## before updating steepest descent direction, store norm squared for
    ## CG coefficients 
    sd_prev_normsq = ccache.criticality_ref[]   # ‖δₖ₋₁‖^2
 
    ## modify carrays.d to be steepest descent direction δₖ
	## store ‖δₖ‖^2 in ccache.criticality_ref[]
    sd_normsq = steepest_descent_direction!(carrays, ccache)    

    @unpack d, Dfx = carrays
    @unpack d_prev, wolfe_constant = step_cache
    if it_index > 1
        ## check wolfe condition at unscaled direction
        fprev_dprev = step_cache.criticality_ref[]  # -max( ⟨∇fᵢ(xₖ₋₁), dₖ₋₁⟩ )
        upper_bound = wolfe_constant * fprev_dprev
        f_dprev = maxdot(Dfx, d_prev)   # max( ⟨∇fᵢ(xₖ), dₖ₋₁⟩ ) = Dₖ(dₖ₋₁)
        lhs = max(
            abs(f_dprev),
            abs(LA.dot(d, d_prev))  # `d` is steepest descent direction atm
        )
        if lhs <= upper_bound
            ## if condition is satisfied, use direction dₖ = θδ + βdₖ₋₁
            denom = fprev_dprev 	# -Dₖ₋₁(dₖ₋₁)
            β = sd_normsq / denom 	# Dₖ(δₖ) / Dₖ₋₁(dₖ₋₁)
            θ = (f_dprev + fprev_dprev) / denom # (Dₖ(dₖ₋₁)-Dₖ₋₁(dₖ₋₁))/(-Dₖ₋₁(dₖ₋₁))
            d .*= θ
            d .+= β .* d_prev
        end		
    end

    ## before scaling, store data for next iteration
    step_cache.criticality_ref[] = abs(maxdot(Dfx, d))
    step_cache.d_prev .= d

    ## compute a stepsize, scale `d`, set `xd .= x .+ d` and values `fxd`
    @unpack sz_cache = ccache
    return stepsize!(carrays, sz_cache, mop; kwargs...)
end

# ╔═╡ bea10146-71ed-416d-9a1b-2ad25108a77b
md"### Fletcher Reeves Fractional LP 1"

# ╔═╡ 01e2cda3-5b22-4546-adb5-3a35ba07ddf6
Base.@kwdef struct FletcherReevesFractionalLP1{
    F<:Real,
    sz_ruleType
} <: AbstractStepRule
    sz_rule :: sz_ruleType = ArmijoBacktracking(; is_modified=Val{true}())
    critval_mode :: Union{Val{:sd}, Val{:cg}} = Val{:sd}()
    constant :: F = 1 + 1e-3
end

# ╔═╡ caefad63-308c-447a-8d47-93a8921afd81
struct FletcherReevesFractionalLP1Cache{
    F<:AbstractFloat,
    ccache_Type
} <: AbstractCGCache
    ccache :: ccache_Type
    critval_mode :: Union{Val{:sd}, Val{:cg}}
    criticality_ref :: Base.RefValue{F}
    constant :: F

    d_prev :: Vector{F}

	Dprev_dprev :: Vector{F}
end

# ╔═╡ 129d195f-531b-44ef-9e8f-f8b79f721ab6
function fr_fractionallp_optindex(Dfx, d_prev, d)
	## Find minimizing index `w` for 
	## ∇f_w(xₖ)ᵀ dₖ₋₁ / ∇f_w(xₖ)ᵀ δₖ
	opt_val = Inf
	w_g_dprev = NaN     # optimal ∇f_w(xₖ)ᵀ dₖ₋₁ 
	w_g_sd = NaN        # optimal ∇f_w(xₖ)ᵀ δₖ
	w = 0
	for (_w, g) in enumerate(eachrow(Dfx))
		g_dprev = LA.dot(g, d_prev)	# ∇f(xₖ)ᵀ dₖ₋₁
		g_sd = LA.dot(g, d)			# ∇f(xₖ)ᵀ δₖ
		_opt_val = g_dprev / g_sd
		if _opt_val < opt_val
			opt_val = _opt_val
			w_g_dprev = g_dprev
			w_g_sd = g_sd
			w = _w
		end
	end
	return w, w_g_dprev, w_g_sd
end

# ╔═╡ fb30b743-284a-4698-a348-9443d5a558cb
function step!(
    carrays, step_cache::FletcherReevesFractionalLP1Cache, mop, it_index; 
    kwargs...
)
    @unpack ccache = step_cache

    ## modify carrays.d to be steepest descent direction
	## store ‖δₖ‖^2 in ccache.criticality_ref[]
    steepest_descent_direction!(carrays, ccache)

    @unpack d, Dfx = carrays
    @unpack d_prev, Dprev_dprev, constant = step_cache

    if it_index > 1
        ## Find optimal index `w`
		(w, w_g_dprev, w_g_sd) = fr_fractionallp_optindex(Dfx, d_prev, d)
        ## use index `w` to define `θ` and `β`
        ### retrieve terms
        w_gprev_dprev = Dprev_dprev[w]      # ∇f_w(xₖ₋₁)ᵀ dₖ₋₁ 
        
        denom = w_gprev_dprev
        θ = (constant * w_gprev_dprev - w_g_dprev) / denom
        β = w_g_sd / denom

        ## set CG direction dₖ = θ δₖ + β dₖ₋₁
        d .*= θ
        d .+= β * d_prev
	else
		d .*= constant
    end

    ## before scaling, store data for next iteration
    step_cache.criticality_ref[] = abs(maxdot(Dfx, d))
    d_prev .= d
    LA.mul!(Dprev_dprev, Dfx, d)

    ## compute a stepsize, scale `d`, set `xd .= x .+ d` and values `fxd`
    @unpack sz_cache = ccache
    return stepsize!(carrays, sz_cache, mop; kwargs...)
end

# ╔═╡ 3c11e6ce-d3c1-4e3c-af5e-cbd465c9b314
md"### Fletcher Reeves Fractional LP 2"

# ╔═╡ 50da11b2-8e61-4103-8bc3-26ce86e6a573
Base.@kwdef struct FletcherReevesFractionalLP2{
    F<:Real,
    sz_ruleType
} <: AbstractStepRule
    sz_rule :: sz_ruleType = ArmijoBacktracking(; is_modified=Val{true}())
    critval_mode :: Union{Val{:sd}, Val{:cg}} = Val{:sd}()
    constant :: F = 1 + 1e-3
end

# ╔═╡ b12e735e-8411-4c4b-b0de-b5d099fddc42
struct FletcherReevesFractionalLP2Cache{
    F<:AbstractFloat,
    ccache_Type
} <: AbstractCGCache
    ccache :: ccache_Type
    critval_mode :: Union{Val{:sd}, Val{:cg}}
    criticality_ref :: Base.RefValue{F}
    constant :: F

    d_prev :: Vector{F}

	Dprev_dprev :: Vector{F}
	Dprev_sdprev :: Vector{F}
	tmp_vec :: Vector{F}
end

# ╔═╡ 1d40ddef-d62e-4edb-b041-521c5c07fd05
function step!(
    carrays, step_cache::FletcherReevesFractionalLP2Cache, mop, it_index; 
    kwargs...
)
    @unpack ccache = step_cache

    ## modify carrays.d to be steepest descent direction
	## store ‖δₖ‖^2 in ccache.criticality_ref[]
    steepest_descent_direction!(carrays, ccache)    

    @unpack d, Dfx = carrays
    @unpack d_prev, Dprev_dprev, Dprev_sdprev, tmp_vec, constant = step_cache

	#LA.mul!(tmp_vec, Dfx, d)	# Dₖ δₖ
	tmp_vec .= d 
	
    if it_index > 1
        ## Find optimal index `w`
		(w, w_g_dprev, w_g_sd) = fr_fractionallp_optindex(Dfx, d_prev, d)
        ## use index `w` to define `θ` and `β` 
        w_gprev_dprev = Dprev_dprev[w]      # ∇f_w(xₖ₋₁)ᵀ dₖ₋₁ 
        w_gprev_sdprev = Dprev_sdprev[w] 	# ∇f_w(xₖ₋₁)ᵀ δₖ₋₁
        
		denom = -constant * w_gprev_sdprev
		c = constant - 1
        θ = (w_g_dprev - w_gprev_dprev - c * w_gprev_sdprev) / denom
        β = - w_g_sd / denom

        ## set CG direction dₖ = θ δₖ + β dₖ₋₁
        d .*= θ
        d .+= β * d_prev
	end

	#@show sign(maxdot(Dfx, d))

    ## before scaling, store data for next iteration
    step_cache.criticality_ref[] = abs(maxdot(Dfx, d))
    d_prev .= d
    LA.mul!(Dprev_dprev, Dfx, d)
	LA.mul!(Dprev_sdprev, Dfx, tmp_vec)
	#Dprev_sdprev .= tmp_vec

    ## compute a stepsize, scale `d`, set `xd .= x .+ d` and values `fxd`
    @unpack sz_cache = ccache
    return stepsize!(carrays, sz_cache, mop; kwargs...)
end

# ╔═╡ c2bf4eab-d078-4cf5-b309-bade8ea84864
md"### Fletcher Reeves Balancing Offset"

# ╔═╡ 36e9f710-310f-41be-8de3-98d406f6e718
Base.@kwdef struct FletcherReevesBalancingOffset{
    F<:Real,
    sz_ruleType
} <: AbstractStepRule
    sz_rule :: sz_ruleType = ArmijoBacktracking(; is_modified=Val{true}())
    critval_mode :: Union{Val{:sd}, Val{:cg}} = Val{:sd}()
    c_gamma :: F = 1.0
	kappa :: F = 1.0
end

# ╔═╡ 91e6b75a-f8da-4a25-8886-898ec1678b75
struct FletcherReevesBalancingOffsetCache{
    F<:AbstractFloat,
    ccache_Type
} <: AbstractCGCache
    ccache :: ccache_Type
    critval_mode :: Union{Val{:sd}, Val{:cg}}
    criticality_ref :: Base.RefValue{F}
    c_gamma :: F
	kappa :: F

    d_prev :: Vector{F}

	D_sd :: Vector{F}
end

# ╔═╡ d8ec968d-f3d5-47c0-bb6d-b07d56e2c0ce
function step!(
    carrays, step_cache::FletcherReevesBalancingOffsetCache, mop, it_index; 
    kwargs...
)
    @unpack ccache = step_cache
 	
	## before updating steepest descent direction, store norm squared for
    ## CG coefficients 
    sd_prev_normsq = ccache.criticality_ref[]   # ‖δₖ₋₁‖^2
 
    ## modify carrays.d to be steepest descent direction
	## store ‖δₖ‖^2 in ccache.criticality_ref[]
    steepest_descent_direction!(carrays, ccache)
	sd_normsq = ccache.criticality_ref[]

    @unpack d, Dfx = carrays
    @unpack d_prev, D_sd, c_gamma, kappa = step_cache

	LA.mul!(D_sd, Dfx, d)	# Dₖ δₖ
	
    if it_index > 1
        ## Find switching index `w`
		D_dprev = maxdot(Dfx, d_prev)
		if D_dprev >= 0
			w_D_sd, w = findmax(D_sd)
		else
			w_D_sd, w = findmin(D_sd)
		end
        ## use index `w` to define `γ`, then `θ` and `β` 
		Γ = (
			( D_dprev * sd_normsq ) - 
			w_D_sd * LA.dot(d, d_prev)
		)
		_γ = (
			( ( -w_D_sd * sd_prev_normsq )/ sd_normsq ) + Γ
		)
		γ = c_gamma / _γ
		#@show γ * Γ <= c_gamma
		β = - γ * w_D_sd
		θ = kappa + γ * D_dprev

        ## set CG direction dₖ = θ δₖ + β dₖ₋₁
        d .*= θ
        d .+= β .* d_prev
	else
		d .*= kappa
	end

    ## before scaling, store data for next iteration
    step_cache.criticality_ref[] = abs(maxdot(Dfx, d))
    d_prev .= d

    ## compute a stepsize, scale `d`, set `xd .= x .+ d` and values `fxd`
    @unpack sz_cache = ccache
    return stepsize!(carrays, sz_cache, mop; kwargs...)
end

# ╔═╡ a3bed194-ea59-4b7e-aa99-0eea6f879c10
md"### Polak-Ribière-Polyak Cone Projection"

# ╔═╡ a7243d4d-c706-412f-9a83-1324b6742c40
Base.@kwdef struct PRPConeProjection{
    sz_rule_Type
} <: AbstractStepRule
    sz_rule :: sz_rule_Type = ArmijoBacktracking(; is_modified=Val{true}())
    critval_mode :: Union{Val{:sd}, Val{:cg}} = Val{:sd}()
end

# ╔═╡ 2577fa49-97bc-43d7-a60b-3885474050e7
struct PRPConeProjectionCache{
    F<:AbstractFloat,
    ccache_Type
} <: AbstractCGCache
    ccache :: ccache_Type
    critval_mode :: Union{Val{:sd}, Val{:cg}}
    criticality_ref :: Base.RefValue{F}

    d_prev :: Vector{F}
    d_orth :: Vector{F}
    d_opt :: Vector{F}
    Dfx_prev :: Matrix{F}
end

# ╔═╡ 6ee68a92-ad08-46a7-bde0-c66b131114b3
"Project `d` on null space of `g` and store in `d⟂`."
function project_on_ker!(
    d⊥::AbstractVector, d::AbstractVector, g::AbstractVector
)
    @assert length(d⊥) == length(d) == length(g)
    g_normsq = sum( g.^2 )
	g_dot_d = LA.dot(g, d)
    d⊥ .= d .- (g_dot_d / g_normsq) .* g
    return d⊥
end

# ╔═╡ c5be6137-f1e5-4dba-be37-ddbe23c3fd1d
function step!(
    carrays, step_cache::PRPConeProjectionCache, mop, it_index; 
    kwargs...
)
    @unpack ccache = step_cache
    ## before updating steepest descent direction, store norm squared for
    ## denominator in CG coefficients 
    sd_prev_normsq = ccache.criticality_ref[]   # -𝔣(δₖ₋₁, xₖ₋₁) = ‖δₖ₋₁‖^2
    
    ## modify carrays.d to be steepest descent direction
    steepest_descent_direction!(carrays, ccache)    # store ‖δₖ‖^2 in ccache.criticality_ref[]

    @unpack d, Dfx = carrays
    @unpack d_prev, d_opt, d_orth, Dfx_prev = step_cache
   
    if it_index > 1
        sd_normsq = ccache.criticality_ref[]    # -𝔣(δₖ, xₖ)
        g_prev_sd = maxdot(Dfx_prev, d)         # 𝔣(δₖ, xₖ₋₁)
        β = (g_prev_sd + sd_normsq) / sd_prev_normsq

        d_prev .*= β
        minimax_outer = Inf
        for (w, gw) in enumerate(eachrow(Dfx))
            project_on_ker!(d_orth, d_prev, gw) # `d_orth` = project dₖ₋₁ onto ∇f(xₖ)ᵀw
            minimax_inner = maxdot(Dfx, d_orth)
            if minimax_inner < minimax_outer
                minimax_outer = minimax_inner
                d_opt .= d_orth
            end
        end
        if minimax_outer <= 0
            ## build CG direction
            d .+= d_opt
        end
    end
    
    ## before scaling, store data for next iteration
    step_cache.criticality_ref[] = abs(maxdot(Dfx, d))
    d_prev .= d

    ## compute a stepsize, scale `d`, set `xd .= x .+ d` and values `fxd`
    @unpack sz_cache = ccache
    return stepsize!(carrays, sz_cache, mop; kwargs...)
end

# ╔═╡ 2efdbe51-0ab9-474f-a749-d04c7341feb7
md"### Polak-Ribière-Polyak Three-Terms"

# ╔═╡ df689303-2d27-4d68-b4a5-05491360f7f0
Base.@kwdef struct PRP3{
    sz_rule_Type
} <: AbstractStepRule
    sz_rule :: sz_rule_Type = ArmijoBacktracking(; is_modified=Val{true}())
    critval_mode :: Union{Val{:sd}, Val{:cg}} = Val{:sd}()
end

# ╔═╡ 18919f7d-b077-465e-a360-c99de01821b3
struct PRP3Cache{
    F<:AbstractFloat,
    ccache_Type
} <: AbstractCGCache
    ccache :: ccache_Type
    critval_mode :: Union{Val{:sd}, Val{:cg}}
    criticality_ref :: Base.RefValue{F}

    d_prev :: Vector{F}
    sd_prev :: Vector{F}
    y :: Vector{F}
end

# ╔═╡ 7c94084c-4519-4e2a-927e-19160726203d
function init_cache(step_rule::PRP3, mop)
    ccache = init_common_cache(step_rule.sz_rule, mop)
    criticality_ref = deepcopy(ccache.criticality_ref)
    @unpack critval_mode = step_rule
    F = float_type(mop)
    d_prev = zeros(F, dim_in(mop))
    sd_prev = similar(d_prev)
    y = similar(d_prev)

    return PRP3Cache(
        ccache, critval_mode, criticality_ref, 
        d_prev, sd_prev, y,
    )
end

# ╔═╡ 6642f8d3-8667-40e7-8dfa-697d1ecbb5d9
function init_common_cache(sz_rule, mop)
    criticality_ref = Ref(convert(float_type(mop), Inf))
    sz_cache = init_cache(sz_rule, mop)
    fw_cache = init_frank_wolfe_cache(float_type(mop), dim_out(mop))
    return CommonStepCache(criticality_ref, sz_cache, fw_cache)
end

# ╔═╡ 1901ba91-4860-45f0-bf87-1f3b3bba15f7
function init_cache(step_rule::SteepestDescentDirection, mop)
    ccache = init_common_cache(step_rule.sz_rule, mop)
    #step_meta = init_step_meta(step_rule.set_metadata, dim_in(mop), 1, float_type(mop))
	step_meta = nothing
    return SteepestDescentDirectionCache(ccache, step_meta)
end

# ╔═╡ 2023970c-27a6-4396-a87c-f9e335da46f3
function init_cache(step_rule::FletcherReevesRestart, mop)
    ccache = init_common_cache(step_rule.sz_rule, mop)
    criticality_ref = deepcopy(ccache.criticality_ref)
    @unpack critval_mode = step_rule
    F = float_type(mop)
    wolfe_constant = convert(F, step_rule.wolfe_constant)
    d_prev = zeros(F, dim_in(mop))
    return FletcherReevesRestartCache(
        ccache, critval_mode, criticality_ref, wolfe_constant, d_prev)
end

# ╔═╡ e01a1988-31b1-4069-b394-d27c760aadb3
function init_cache(step_rule::FletcherReevesFractionalLP1, mop)
    ccache = init_common_cache(step_rule.sz_rule, mop)
    criticality_ref = deepcopy(ccache.criticality_ref)
    @unpack critval_mode = step_rule
    F = float_type(mop)
    constant = convert(F, step_rule.constant)
    d_prev = zeros(F, dim_in(mop))

    Dprev_dprev = zeros(F, dim_out(mop))
    return FletcherReevesFractionalLP1Cache(
        ccache, critval_mode, criticality_ref, constant, d_prev,
        Dprev_dprev
    )
end

# ╔═╡ b4bc8ec5-f141-47e8-b829-d40d59990406
function init_cache(step_rule::FletcherReevesFractionalLP2, mop)
    ccache = init_common_cache(step_rule.sz_rule, mop)
    criticality_ref = deepcopy(ccache.criticality_ref)
    @unpack critval_mode = step_rule
    F = float_type(mop)
    constant = convert(F, step_rule.constant)
    d_prev = zeros(F, dim_in(mop))

    Dprev_dprev = zeros(F, dim_out(mop))
	Dprev_sdprev = zeros(F, dim_out(mop))
	tmp_vec = zeros(F, dim_in(mop))

    return FletcherReevesFractionalLP2Cache(
        ccache, critval_mode, criticality_ref, constant, d_prev,
        Dprev_dprev, Dprev_sdprev, tmp_vec
    )
end

# ╔═╡ 6f2c0b9f-83fe-484a-8176-6bc0847d9ad4
function init_cache(step_rule::FletcherReevesBalancingOffset, mop)
    ccache = init_common_cache(step_rule.sz_rule, mop)
    criticality_ref = deepcopy(ccache.criticality_ref)
    @unpack critval_mode = step_rule
    F = float_type(mop)
    c_gamma = convert(F, step_rule.c_gamma)
	kappa = convert(F, step_rule.kappa)
    d_prev = zeros(F, dim_in(mop))

    D_sd = zeros(F, dim_out(mop))
    return FletcherReevesBalancingOffsetCache(
        ccache, critval_mode, criticality_ref, c_gamma, kappa, d_prev,
        D_sd
    )
end

# ╔═╡ 634c40b5-299f-4b77-8e07-2c5a15a73e06
function init_cache(step_rule::PRPConeProjection, mop)
    ccache = init_common_cache(step_rule.sz_rule, mop)
    criticality_ref = deepcopy(ccache.criticality_ref)
    @unpack critval_mode = step_rule

    F = float_type(mop)
    d_prev = zeros(F, dim_in(mop))
    d_orth = similar(d_prev)
    d_opt = similar(d_prev)

    Dfx_prev = zeros(F, dim_out(mop), dim_in(mop))

    return PRPConeProjectionCache(
        ccache, critval_mode, criticality_ref, 
        d_prev, d_orth, d_opt, Dfx_prev
    )
end

# ╔═╡ 3c1ed4ad-f4ab-4052-974b-f41953f4300a
function step!(
    carrays, step_cache::PRP3Cache, mop, it_index; 
    kwargs...
)
    @unpack ccache = step_cache
    ## before updating steepest descent direction, store norm squared for
    ## denominator in CG coefficients 
    sd_prev_normsq = ccache.criticality_ref[]   # ‖δₖ₋₁‖^2
    
    ## modify carrays.d to be steepest descent direction
    steepest_descent_direction!(carrays, ccache)    # store ‖δₖ‖^2 in ccache.criticality_ref[]

    @unpack d, Dfx = carrays
    @unpack sd_prev, d_prev, y = step_cache
   	#@show d
    if it_index > 1
        ## set difference vector
        ### at this point, `sd_prev` holds δₖ₋₁
        y .= sd_prev .- d     # yₖ = δₖ₋₁ - δₖ

        ## determine (wβ, vβ) by solving discrete minimax problem
        ## min_w max_v ⟨w, ∇f(xₖ)yₖ⟩ ⋅ ⟨v, ∇f(xₖ)dₖ₋₁⟩
        wβ = vβ = 0
        wβ_g_y = NaN    # ⟨wβ, ∇f(xₖ)yₖ⟩

        ## It always holds that
        ## min_w max_v ⟨w, ∇f(xₖ)yₖ⟩ ⋅ ⟨v, ∇f(xₖ)dₖ₋₁⟩
        ## ≥
        ## max_v min_w ⟨w, ∇f(xₖ)yₖ⟩ ⋅ ⟨v, ∇f(xₖ)dₖ₋₁⟩
        ## = 
        ## max_w min_v ⟨v, ∇f(xₖ)yₖ⟩ ⋅ ⟨w, ∇f(xₖ)dₖ₋₁⟩
        ##
        ## determine (wθ, vθ) by solving discrete maximin problem
        ## max_w min_v ⟨v, ∇f(xₖ)yₖ⟩ ⋅ ⟨w, ∇f_w(xₖ)dₖ₋₁⟩
        wθ = vθ = 0
        wθ_g_dprev = NaN    # ⟨wθ, ∇f(xₖ)dₖ₋₁⟩

        minimax_outer = Inf
        maximin_outer = -Inf
        for (w, gw) in enumerate(eachrow(Dfx))
            ## max_v ⟨w, ∇f(xₖ)yₖ⟩ ⋅ ⟨v, ∇f(xₖ)dₖ₋₁⟩
            minimax_inner = -Inf
            vmax = 0
            w_g_y = LA.dot(gw, y)           # ⟨w, ∇f(xₖ)yₖ⟩

            ## min_v ⟨v, ∇f(xₖ)yₖ⟩ ⋅ ⟨w, ∇f_w(xₖ)dₖ₋₁⟩
            maximin_inner = Inf
            vmin = 0
            w_g_dprev = LA.dot(gw, d_prev)   # ⟨w, ∇f_w(xₖ)dₖ₋₁⟩

            for (v, gv) in enumerate(eachrow(Dfx))
                v_g_dprev = LA.dot(gv, d_prev)      # ⟨v, ∇f(xₖ)dₖ₋₁⟩
                _minimax_inner = w_g_y * v_g_dprev  # ⟨w, ∇f(xₖ)yₖ⟩ ⋅ ⟨v, ∇f(xₖ)dₖ₋₁⟩
                if _minimax_inner > minimax_inner
                    minimax_inner = _minimax_inner
                    vmax = v
                end
                v_g_y = LA.dot(gv, y)               # ⟨v, ∇f(xₖ)yₖ⟩
                _maximin_inner = w_g_dprev * v_g_y  # ⟨v, ∇f(xₖ)yₖ⟩ ⋅ ⟨w, ∇f_w(xₖ)dₖ₋₁⟩
                if _maximin_inner < maximin_inner
                    maximin_inner = _maximin_inner
                    vmin = v
                end
            end
            if minimax_inner < minimax_outer
                minimax_outer = minimax_inner
                wβ_g_y = w_g_y
                wβ = w
                vβ = vmax
            end
            if maximin_inner > maximin_outer
                maximin_outer = maximin_inner
                wθ_g_dprev = w_g_dprev
                wθ = w
                vθ = vmin
            end
        end

        ψβ = minimax_outer
        ψθ = maximin_outer
        ## Note: Minimax Theorem holds for discrete problems: ψβ ≥ ψθ

        ## determine balancing coefficients
        αβ, αθ = if ψβ == ψθ
            (1, 1)
        elseif ψθ < 0 && ψβ >= 0 || ψθ == 0 && ψβ > 0
            ### sign switch / restart
            (0, 0)
        elseif ψθ > 0
            ### both positive, 0 < ψθ ≤ ψβ
            ## shrink larger factor
            (ψθ/ψβ, 1)
        else
            ### both negative, ψθ ≤ ψβ < 0
            ### grow smaller factor
            (1, ψβ/ψθ)
        end

        ## finally, set coefficients
        β = αβ * wβ_g_y / sd_prev_normsq
        θ = αθ * wθ_g_dprev / sd_prev_normsq

    	## before modifying `d`, store steepest descent direction for next iteration
    	sd_prev .= d

	    ## build CG direction
	    d .+= β .* d_prev
	    d .-= θ .* y
	else
		sd_prev .= d
	end
	
    ## before scaling, store data for next iteration
    step_cache.criticality_ref[] = abs(maxdot(Dfx, d))
    d_prev .= d

    ## compute a stepsize, scale `d`, set `xd .= x .+ d` and values `fxd`
    @unpack sz_cache = ccache
    return stepsize!(carrays, sz_cache, mop; kwargs...)
end

# ╔═╡ 97e33422-6d1b-4bc4-a86b-e50536d5537a
md"# Utils"

# ╔═╡ 490b7c22-1a07-4506-b4bb-1bdb6e191ca1
md"## GatheringCallback"

# ╔═╡ 594e804a-9589-4787-8612-1c51b1d324c2
struct GatheringCallback{F, step_meta_Type}
    x :: Vector{Vector{F}}
    fx :: Vector{Vector{F}}
    critvals :: Vector{F}
    step_meta :: step_meta_Type

	function GatheringCallback(::Type{F}) where F<:Number
	    x = Vector{F}[]
	    fx = Vector{F}[]
	    critvals = F[]
	    step_meta = nothing
	    return new{F, Nothing}(x, fx, critvals, step_meta)
	end
end

# ╔═╡ 8ac3435d-1375-4e0e-b2dd-d48388542e3b
function Base.empty!(cback::GatheringCallback)
    @unpack x, fx, critvals, step_meta = cback
    empty!(x)
    empty!(fx)
    empty!(critvals)
    if isa(step_meta, AbstractVector)
        empty!(step_meta)
    end
	return cback
end

# ╔═╡ 5ac94c34-d0f7-4b4d-b33c-3f513d7a510d
function initialize_callback(callback::GatheringCallback, carrays, mop, step_cache)
	empty!(callback)
    return callback
end

# ╔═╡ 055ca4a5-a451-4cd5-ae36-2879a3d0fded
function exec_callback(callback::GatheringCallback, it_index, carrays, mop, step_cache, stop_code)
	
    push!(callback.x, copy(carrays.x))
    push!(callback.fx, copy(carrays.fx))
    push!(callback.critvals, criticality(carrays, step_cache))
    
    return nothing
end

# ╔═╡ a4708b12-9bd8-4eb5-a8cc-004807daaa37
function optimize_after_init(
    carrays, 
	mop, 
	step_cache, 
	callback;
    max_iter, 
    crit_tol_abs,
    x_tol_rel, x_tol_abs,
    fx_tol_rel, fx_tol_abs,
    log_level 
)
    @unpack x, fx, Dfx, d, xd, fxd, y = carrays

	## update `fx` as well as `Dfx`
    stop_code = objectives_and_jac!(fx, Dfx, mop, x)
    it_index = 1
	callback_called = false
    while true
        callback_called = false
		@ignorebreak stop_code 	# exhausted budget even before first iteration ?

		## check number of iterations
        @ignorebreak stop_code = it_index > max_iter ? STOP_MAX_ITER : nothing
        
        @logmsg log_level """#========================================#
        Iteration $(it_index)
        x  = $(pretty_row_vec(x))
        fx = $(pretty_row_vec(fx))"""

		## compute step `d`, also set `xd` and `fxd`!
		### `@ignorebreak` to be strict with budget
        @ignorebreak stop_code = step!(
            carrays, step_cache, mop, it_index;
            crit_tol_abs, x_tol_rel, x_tol_abs, fx_tol_rel, fx_tol_abs
        )
        ## derive criticality
        critval = criticality(carrays, step_cache)

		## update displacement vector
		@. y = fxd - fx
       
      	## before updates, do callback, enabling break before Jacobian call
		## also, this way, callback has access to both current and next values
        @ignorebreak stop_code = begin
            callback_called = true
            exec_callback(callback, it_index, carrays, mop, step_cache, stop_code)
        end
		 
        ## compute test values **before** updating `x`, `fx` etc.
		x_norm = LA.norm(x, Inf)
        fx_norm = LA.norm(fx, Inf)
		x_change = LA.norm(d, Inf)
        fx_change = LA.norm(y, Inf)
		
		@logmsg log_level """\n
        critval   = $(critval)
        x_change  = $(x_change)
        fx_change = $(fx_change)"""

		## update iterates and Jacobian
		x .= xd
        fx .= fxd
		@ignorebreak stop_code = jac!(Dfx, mop, x)
		     
        ## check other stopping criteria
		@ignorebreak stop_code = critval < crit_tol_abs ? STOP_CRIT_TOL_ABS : nothing
		
        @ignorebreak stop_code = x_change < x_tol_abs ? STOP_X_TOL_ABS : nothing
        @ignorebreak stop_code = x_change < x_tol_rel * x_norm ?  STOP_X_TOL_REL : nothing

        @ignorebreak stop_code = fx_change < fx_tol_abs ? STOP_FX_TOL_ABS : nothing
        @ignorebreak stop_code = fx_change < x_tol_rel * fx_norm ? STOP_FX_TOL_REL : nothing

		## proceed to next iteration      
        it_index += 1
    end
    if !callback_called
        exec_callback(callback, it_index, carrays, mop, step_cache, stop_code)
    end

    return (; 
        x, fx, carrays, step_cache, stop_code, callback,
		num_its = it_index - 1, 	# number of finished iterations
        num_func_calls = num_calls_objectives(mop),
        num_grad_calls = num_calls_jac(mop)
    )
end

# ╔═╡ 1c886e8d-1551-4889-b662-3998754f8e90
function optimize(
    x0::AbstractVector{<:Real}, 
	mop::MOP,
    ;
	callback = DefaultCallback(),
    step_rule = SteepestDescentDirection(),
    max_iter = 500,
    max_func_calls = nothing,
    max_grad_calls = nothing,
    crit_tol_abs = 1e-10,    # Optim.jl has 1e-8
    x_tol_rel = 1e-9,
    x_tol_abs = 0,
    fx_tol_rel = 0,
    fx_tol_abs = 0,
    log_level = Info
)
    ## some sanity checks
    @assert length(x0) == dim_in(mop) "Variable vector has wrong length."
    @assert dim_in(mop) > 0 "Variable dimension must be positive."
    @assert dim_out(mop) > 0 "Output dimension must be positive."

    ## set `max_iter` to a sensible value
    if isnothing(max_iter) || max_iter < 0
        max_iter = Inf
    end

	mop = wrap_mop(mop)
	## restrict number of function calls
	if max_func_calls isa Number && max_func_calls >= 0
		max_num_calls_objectives!(mop, ceil(Int, max_func_calls))
	end
	if max_grad_calls isa Number && max_grad_calls >= 0
		max_num_calls_jac!(mop, ceil(Int, max_grad_calls))
	end
	
    ## initialize common arrays and set initial iteration site
    carrays = initialize_common_arrays(mop)
    @unpack x, fx, Dfx = carrays
    copyto!(x, x0)

    ## prepare cache for steps
    step_cache = init_cache(step_rule, mop)

    ## prepare_callback
    callback = initialize_callback(callback, carrays, mop, step_cache)

    return optimize_after_init(
        carrays, mop, step_cache, callback;
        max_iter, 
        crit_tol_abs,
        x_tol_rel, x_tol_abs,
        fx_tol_rel, fx_tol_abs,
        log_level 
    )
end

# ╔═╡ 0300b3df-e65c-43e6-83bb-01ef7235f76e
md"# Tests"

# ╔═╡ ac5ae896-bc8e-414a-b47f-924fa201be91
# ╠═╡ skip_as_script = true
#=╠═╡
testdata = let
	f_rb(x1, x2, a, b) = b * (x2 - x1^2)^2 + (a - x1)^2
	df_rb(x1, x2, a, b) = [
	    -4*b*(x2-x1^2)*x1 - 2*(a-x1),
	    2*b*(x2-x1^2)
	]
	a1 = 1.0
	a2 = 2.0
	b1 = b2 = 100

	mop = MOP(;
		dim_in = 2,
		dim_out = 2,
		objectives! = function (y, x)
			y[1] = f_rb(x[1], x[2], a1, b1)
			y[2] = f_rb(x[1], x[2], a2, b2)
			nothing
		end,
		jac! = function (Dy, x)
			Dy[1, :] .= df_rb(x[1], x[2], a1, b1)
			Dy[2, :] .= df_rb(x[1], x[2], a2, b2)
			nothing
		end
	)
	(; a1, b1, mop)
end
  ╠═╡ =#

# ╔═╡ 4d5efea3-6363-4144-9cc3-7225d94a9011
# ╠═╡ skip_as_script = true
#=╠═╡
cback_sd, cback_cg = let
	@unpack mop = testdata
	@show x0 = -4 .+ 8 .* rand(2)
	# x0 = [3.1, -2.9] 	# weird with FractionalLinearLP1
	step_rule1 = SteepestDescentDirection(;
		sz_rule = FixedStepsize()
	)
	step_rule2 = SteepestDescentDirection(;
		sz_rule = ArmijoBacktracking(;
			is_modified = Val(false)
		)
	)
	step_rule3 = FletcherReevesRestart(;
		sz_rule = ArmijoBacktracking(;
			is_modified = Val(true)
		)
	)
	step_rule4 = FletcherReevesFractionalLP1(;
		sz_rule = ArmijoBacktracking(;
			is_modified = Val(true)
		)
	)
	step_rule5 = FletcherReevesFractionalLP2(;
		sz_rule = ArmijoBacktracking(;
			is_modified = Val(true)
		)
	)
	step_rule6 = FletcherReevesBalancingOffset(;
		c_gamma = 2.0,
		sz_rule = ArmijoBacktracking(;
			is_modified = Val(true)
		)
	)
	step_rule7 = PRPConeProjection(;
		sz_rule = ArmijoBacktracking(;
			is_modified = Val(true)
		)
	)
	step_rule8 = PRP3(;
		sz_rule = ArmijoBacktracking(;
			is_modified = Val(true)
		)
	)
	res_tuple_sd = optimize(
		x0, mop::MOP
    	;
		max_iter = 50,
	    step_rule = step_rule2,
		callback = GatheringCallback(Float64),
	    x_tol_rel = 1e-9,
		log_level = LogLevel(-1001)
	)
	res_tuple_cg = optimize(
		x0, mop::MOP
    	;
		max_iter = 50,
	    step_rule = step_rule8,
		callback = GatheringCallback(Float64),
	    x_tol_rel = 1e-9,
		log_level = LogLevel(-1001)
	)
	res_tuple_sd.callback, res_tuple_cg.callback
end
  ╠═╡ =#

# ╔═╡ 723a0f8f-82ec-4f0a-8c2a-58c43bf1cfb7
# ╠═╡ skip_as_script = true
#=╠═╡
let a1 = 1.0, a2 = 2.0;
	fig = Figure()
	ax = Axis(fig[1,1])
	
	pset_x1 = range(a1, a2, 100)
	pset_x2 = pset_x1.^2
	pset = vcat(pset_x1', pset_x2')
	lines!(ax, pset; color = :green)

	X_sd = reduce(hcat, cback_sd.x)
	X_cg = reduce(hcat, cback_cg.x)
	scatterlines!(ax, X_sd; color = :red)
	scatterlines!(ax, X_cg; color = :blue)
	
	fig
end
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
UnPack = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
WGLMakie = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"

[compat]
PlutoUI = "~0.7.60"
UnPack = "~1.0.2"
WGLMakie = "~0.10.17"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "4da0422d44e03fede575e1b07cc9575248d2469f"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bonito]]
deps = ["Base64", "CodecZlib", "Colors", "Dates", "Deno_jll", "HTTP", "Hyperscript", "LinearAlgebra", "Markdown", "MsgPack", "Observables", "RelocatableFolders", "SHA", "Sockets", "Tables", "ThreadPools", "URIs", "UUIDs", "WidgetsBase"]
git-tree-sha1 = "262f58917d5d9644d16ec6f53480e11a6e128db2"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f8"
version = "4.0.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+2"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "ea32b83ca4fefa1768dc84e504cc0a94fb1ab8d1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "e1371a23fd9816080c828d0ce04373857fe73d33"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.3"

[[deps.Deno_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cd6756e833c377e0ce9cd63fb97689a255f12323"
uuid = "04572ae6-984a-583e-9378-9577a1c2574d"
version = "1.33.4+0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3101c32aab536e7a27b1763c0797dba151b899ad"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.113"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e51db81749b0777b2147fbe7b783ee79045b8e99"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.4+1"

[[deps.Extents]]
git-tree-sha1 = "81023caa0021a41712685887db1fc03db26f41f5"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.4"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "8cc47f299902e13f90405ddb5bf87e5d474c0d38"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "6.1.2+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "7878ff7172a8e6beedd1dea14bd27c3c6340d361"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.22"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "d52e255138ac21be31fa633200b65e4e71d26802"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.6"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "59107c179a586f0fe667024c5eb7033e81333271"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.2"

[[deps.GeoInterface]]
deps = ["Extents", "GeoFormatTypes"]
git-tree-sha1 = "826b4fd69438d9ce4d2b19de6bc2f970f45f0f88"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.8"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "b62f2b2d76cee0d61a2ef2b3118cd2a3215d3134"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.11"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0224cce99284d997f6880a42ef715a37c99338d1"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.2+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "48b5d4c75b2c9078ead62e345966fa51a25c05ad"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.2+1"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "dc6bed05c15523624909b3953686c5f5ffa10adc"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "6c22309e9a356ac1ebc5c8a217045f9bae6f8d9a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.13"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "b1c2585431c382e3fe5805874bda6aea90a95de9"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.25"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "LinearAlgebra", "MacroTools", "RoundingEmulator"]
git-tree-sha1 = "24c095b1ec7ee58b936985d31d5df92f9b9cfebb"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.19"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

    [deps.IntervalArithmetic.weakdeps]
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "25ee0be4d43d0269027024d75a24c24d6c6e590c"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.4+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "36bdbc52f13a7d1dcb0f3cd694e01677a515655b"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+1"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c6ce1e19f3aec9b59186bdf06cdf3c4fc5f5f3e6"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.50.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "61dfdba58e585066d8bce214c5a51eaa0539f269"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "84eef7acd508ee5b3e956a2ae51b05024181dee0"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.2+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "b404131d06f7886402758c9ce2214b636eb4d54a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "edbf5309f9ddf1cab25afc344b1e8150b7c832f9"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.2+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "260d6e1ac8abcebd939029e6eedeba4e3870f13a"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.21.17"

[[deps.MakieCore]]
deps = ["ColorTypes", "GeometryBasics", "IntervalSets", "Observables"]
git-tree-sha1 = "b774d0563bc332f64d136d50d0420a195d9bdcc6"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.8.11"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "f45c8916e8385976e1ccd055c9874560c257ab13"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.2"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "39d000d9c33706b8364817d8894fae1548f40295"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.2"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "52af86e35dd1b177d051b12681e1c581f53c281b"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "79123bc60c5507f035e6d1d9e563bb2971954ec8"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "777657803913ffc7e8cc20f0fd04b634f871af8f"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.8"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "9537ef82c42cdd8c5d443cbc359110cbb36bae10"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.21"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadPools]]
deps = ["Printf", "RecipesBase", "Statistics"]
git-tree-sha1 = "50cb5f85d5646bc1422aa0238aa5bfca99ca9ae7"
uuid = "b189fb0b-2eb5-4ed4-bc0c-d34c51242431"
version = "2.1.1"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "0248b1b2210285652fbc67fd6ced9bf0394bcfec"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.1"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "01915bfcd62be15329c9a07235447a89d588327c"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.21.1"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.WGLMakie]]
deps = ["Bonito", "Colors", "FileIO", "FreeTypeAbstraction", "GeometryBasics", "Hyperscript", "LinearAlgebra", "Makie", "Observables", "PNGFiles", "PrecompileTools", "RelocatableFolders", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "db71caa2e1ac6b3f806333c9de32393ed75d60e6"
uuid = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"
version = "0.10.17"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WidgetsBase]]
deps = ["Observables"]
git-tree-sha1 = "30a1d631eb06e8c868c559599f915a62d55c2601"
uuid = "eead4739-05f7-45a1-878c-cee36b57321c"
version = "0.1.4"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "15e637a697345f6743674f1322beefbc5dcd5cfc"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+1"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2b0e27d52ec9d8d483e2ca0b72b3cb1a8df5c27a"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+1"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "02054ee01980c90297412e4c809c8694d7323af3"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+1"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+1"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee57a273563e273f0f53275101cd41a8153517a"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+1"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+1"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b9ead2d2bdb27330545eb14234a2e300da61232e"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b70c870239dc3d7bc094eb2d6be9b73d27bef280"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "7dfa0fd9c783d3d0cc43ea1af53d69ba45c447df"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "ccbb625a89ec6195856a50aa2b668a5c08712c94"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.4.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "35976a1216d6c066ea32cba2150c4fa682b276fc"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.0+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc541bb19ed5b0ede95581fb2e41ecf179527d2"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.6.0+0"
"""

# ╔═╡ Cell order:
# ╠═1d45f603-f9ca-4815-b94d-a7674c2fbb4b
# ╠═737f0c57-6146-4677-98b0-251e412a7ad8
# ╠═fea950e4-29c6-41ed-be89-a58b222f7973
# ╠═dfefc188-e6de-42fe-b782-7a4011fc1073
# ╟─4606c253-f824-4468-8c08-72ab120b720d
# ╟─c6beceae-b713-11ef-0890-67d740e56c6c
# ╟─07418323-ffe7-4053-8498-7a7bd9693917
# ╟─423e5a3a-8869-4f8e-b42c-3e96f405aed8
# ╟─1c8552a3-2ace-4519-8287-5cf43b23f68b
# ╠═2c24581f-af6f-4209-af8b-448f32c2fea7
# ╟─b56c13ea-0ea3-4865-9e1e-526cf07b8081
# ╠═a1c06c57-6c3a-4ba9-85f5-ec1f1c4ef6e9
# ╟─8ba05829-a74a-4fd4-9310-87e9f147519a
# ╠═48534191-f29a-45b4-a4c9-2e8854deb839
# ╠═189d0a44-6671-4521-8d7e-c5584a6f8375
# ╟─d22d376b-e8b3-4a3c-9afa-3b43575bc1ea
# ╟─f1dd1c2d-d521-4f2d-90bd-952379f3427d
# ╠═6db3895b-dae5-4d5d-a539-2667f932cf48
# ╠═3b38f68b-0a07-4898-a7b3-25affa633ed8
# ╟─cd6e18ca-f659-4100-a47d-409262660384
# ╠═955b2796-8279-4831-b12b-8fa091721dd5
# ╠═e29f3212-71ce-43f9-96ed-bc1597f2a88c
# ╠═5d741ef6-b695-4915-8da6-1a50d943e875
# ╟─cc255919-7bcd-48c3-8e3b-c86ec365c977
# ╠═0032a05a-3989-4ea0-ba2c-de683f0cbe4b
# ╠═0277713e-bb2f-43d9-837e-583c944324fa
# ╠═3ca6b8c2-94b2-4c71-890b-421b987bbf11
# ╠═8bcb7058-b529-4009-9874-db494183b7d2
# ╠═c9ad3d97-330d-4fd7-a468-656f5cb578a5
# ╠═b972d504-3518-4b6e-8f0e-afd595f62a32
# ╠═69ce9e0f-d336-4c55-a7d4-86ec941924cd
# ╠═d9fee57d-ee9f-4ed7-aa85-9e61472ff4cd
# ╟─dd82f281-a10b-44ba-821a-a56f2923ae8b
# ╠═35ff7b4a-9bc8-4e26-8371-736eb04e4fba
# ╠═9c09fc70-c270-4515-a2eb-241650f2c651
# ╠═9cf361e5-57f3-48e5-9405-a2c6f5df8b10
# ╠═05cd3f07-2e41-4ebf-9cdf-bc32248c0aa5
# ╠═52c01c37-130f-44d3-b9e2-b15af18cc251
# ╠═370b2e94-8628-4582-9a87-7d7e4bfa32ab
# ╠═611df4e2-3c46-4ded-a222-9d3cbc46f1b7
# ╠═a58d490c-2164-410d-ac5a-241e4b8130ba
# ╠═2f3fe957-9f8b-47cd-9f3c-6db914956fa1
# ╠═7e3406d3-8252-4b4e-bd9a-b1edb11a0d21
# ╟─a9b50868-2880-47df-904e-aed3891f73d4
# ╟─03d32a31-9a60-4556-a2e6-b048ab1160bc
# ╟─6773003d-51e6-43b8-bd05-06089bda2872
# ╠═1a7c4eb1-2981-4a01-8526-37fb9a4ea05a
# ╟─ad0b0559-41eb-49a1-a6ab-455201160975
# ╠═14b9e1a0-435c-41af-9368-41ea8a899c97
# ╠═a00b4a6a-b93d-4859-abfb-be43c4d3558c
# ╟─ffb6dc4d-4f02-4127-ab01-f858ac12b805
# ╟─06bb50f9-18b5-402b-ae2b-daae9ff96b83
# ╠═2ff5e653-86b8-473b-9e35-b4091a298e89
# ╟─38103ab8-b876-4ecf-8830-41249dd41904
# ╠═e557a091-18ab-4502-a419-c1cd14bc876b
# ╟─36262f76-15de-4920-b5d3-bea05db849bd
# ╟─7fc47384-31b8-4fcf-b7f1-2ca9d8d1393c
# ╠═460b2c61-6e9c-43a2-aae6-92f1171c8660
# ╠═120acaa5-b79f-4914-9c8c-c47a0d844c83
# ╟─efbb21f2-a445-43c7-b5ea-213eed77c3da
# ╠═651abe09-9eeb-4f90-851e-23a0e32a6c96
# ╟─8f4507d2-1d12-42f5-953b-b8b5da8e30a5
# ╠═c4ca79fd-00b0-4d8c-8699-73b4dbc54f41
# ╟─4e515bed-d72f-4865-a334-4b0c16768893
# ╠═20872fb5-f12c-437b-821f-62b394987f07
# ╟─2ccf8082-6427-4d13-b155-f340c1025d9f
# ╟─70e74fd5-2242-4fc7-9dae-59e48f1c77e0
# ╠═86ab5eb1-9c50-48fa-8fd3-575cc2e50eec
# ╠═8d8caa14-d97c-4658-8e98-edbbcac2dca6
# ╟─ac6d8197-fa82-43d4-9f06-0422d73e62c5
# ╠═a837dc91-6a85-4a7a-8cdc-70f8b9b1d869
# ╟─d20e013f-f1c3-46b3-8cf6-4e07d202ae84
# ╠═c7c62a50-4d7c-41b9-9905-2a5f6da55fab
# ╟─9a0646d2-1f30-4db7-9d3f-2ba3b7cf10a1
# ╟─a26663d2-730b-46f0-bedb-cc692e68516c
# ╠═95e2caeb-fe58-424e-ac3a-7a01c86d36fe
# ╠═c0116a05-eef2-41a7-aed9-318188cb694a
# ╠═5d8267cc-27d0-438f-b326-8a8583b54f4b
# ╟─c813dbbd-e7c3-4308-9a15-5055bf9b14a7
# ╠═1c886e8d-1551-4889-b662-3998754f8e90
# ╠═a4708b12-9bd8-4eb5-a8cc-004807daaa37
# ╟─beb2c714-4037-45d7-875a-34e89bf6deab
# ╟─192cedc1-f52d-49c4-aca3-29b2920e5a75
# ╟─2cd698d7-dc83-4095-ac65-e63b563e965d
# ╠═1fbad73b-046d-417f-b7ef-22287fb7b5d9
# ╠═c432d8e6-6148-44e1-bf29-73f172d2f1ec
# ╟─d6d001e3-6641-45c5-b295-1246d3397336
# ╠═df3d1851-9a1f-489e-9f1e-b3baaa24a914
# ╟─c0642afa-96f1-49d4-8cae-16d34c5b62e3
# ╠═4ae8c923-8af7-4d71-9354-1be96b5d0cfc
# ╟─a406b503-262d-4773-960e-4d13fce5ff0d
# ╠═082a8d61-ac69-4ed7-addc-f8e951f80fed
# ╠═cf720d41-d841-4ecd-a2f7-15d53b22a35d
# ╠═a0893f2c-1342-4099-a974-f883808f45df
# ╠═16bdabcc-b652-4d64-b350-da65316d973e
# ╟─cb7497f7-1b31-4549-a6f4-019a90443ed1
# ╟─3a361a85-8469-4677-b70f-da49019035e4
# ╠═f03796a7-3ff1-477c-a4a4-8a099d6b92ce
# ╟─76794911-fecb-44b8-8459-d52939145e55
# ╠═831a2337-9c28-4b93-8871-b80b3b5dee77
# ╠═6beffea4-098b-4f2e-9efd-87335321027b
# ╠═14cf9e93-8769-4de5-a076-4e8237e03e1f
# ╟─4dba2ef2-1c12-4e54-9557-0c50a98c6d74
# ╠═92e74936-9522-480d-baf6-6ff1836e2c43
# ╠═f27c17a3-dc89-4041-9555-940dbfdefcdc
# ╠═f79a2daa-1c83-453c-9d8f-2933b9aee56b
# ╠═0a224393-3dad-415d-a009-4d29a5bcf0af
# ╠═46d1e46b-3d2a-4b3f-8851-5995816a54c7
# ╠═36cb9a2c-1df8-4982-9ee0-50a5a687d76a
# ╠═1ba108a2-65cc-4bbe-a6bd-63fc1c7e0531
# ╠═f0abfe44-86d1-406c-aed8-a4d1bb6e89b0
# ╠═5c1be58e-98aa-4367-8a1d-3f96dc3ccbd9
# ╠═35c34c50-ab63-4493-9a8a-24ac1952540f
# ╠═91afbddf-2aba-4428-be04-e06020ad694b
# ╟─0c49efa6-cd89-493e-8b1a-85571aae2761
# ╟─d7ba86be-ad06-490e-bac8-908fd0638b29
# ╟─8bae3396-36e7-4532-a946-ebbbd9b5a5d4
# ╟─492ff20d-64af-49ad-8a0f-9799fee41051
# ╠═e862a437-6ecf-46f8-adce-916b0215b0e2
# ╟─49f21216-679d-4f39-b9e4-089ac5b7ebb0
# ╠═de7628ff-ce1d-451f-8eec-0e956a42e4eb
# ╟─589768dd-331c-4da0-a11e-d0c2c0f40984
# ╠═60e85226-cdfe-4e57-b489-124bc84c7120
# ╠═6642f8d3-8667-40e7-8dfa-697d1ecbb5d9
# ╟─bd8a0cb0-926a-4824-9a32-ff8127eeff61
# ╠═a3571fc6-12ff-4827-8b99-7e2df8bcd187
# ╠═e4148017-12b9-4084-a88a-94c4f325ddf2
# ╠═893a3e63-72e1-46a3-bc3c-9eab65d347f5
# ╟─9bb1d034-58f2-463b-aa7b-f0630ab52f5d
# ╠═de5cff65-cf87-4ff6-bc98-56278c5e8539
# ╠═1901ba91-4860-45f0-bf87-1f3b3bba15f7
# ╠═707a6e6e-d956-4261-8db1-83987cab8a1b
# ╠═3fd9a9ef-b268-4b22-b868-e7858152ae26
# ╟─e692c5ca-57d5-494a-9469-006109c0ca02
# ╠═7723b535-3cc4-44b3-8b21-c0f0490b5ebf
# ╠═91a999db-6428-43c1-975b-7859b0af6c00
# ╠═9ea12970-eca2-47ff-8dee-90c222af5c9e
# ╠═f5388f8d-4013-491d-9c24-58d5c19cf9a5
# ╠═92d896fe-7066-4fc7-a2fd-cf91ccc9a466
# ╠═37cfa713-1fc5-4878-baa1-2b31a348a505
# ╠═44d26cf8-9f54-4604-8a4d-c85a65ce2d5b
# ╠═2023970c-27a6-4396-a87c-f9e335da46f3
# ╠═091bf95f-b0e6-4ff6-b1ac-4b765eacffb3
# ╟─bea10146-71ed-416d-9a1b-2ad25108a77b
# ╠═01e2cda3-5b22-4546-adb5-3a35ba07ddf6
# ╠═caefad63-308c-447a-8d47-93a8921afd81
# ╠═e01a1988-31b1-4069-b394-d27c760aadb3
# ╠═fb30b743-284a-4698-a348-9443d5a558cb
# ╠═129d195f-531b-44ef-9e8f-f8b79f721ab6
# ╟─3c11e6ce-d3c1-4e3c-af5e-cbd465c9b314
# ╠═50da11b2-8e61-4103-8bc3-26ce86e6a573
# ╠═b12e735e-8411-4c4b-b0de-b5d099fddc42
# ╠═b4bc8ec5-f141-47e8-b829-d40d59990406
# ╠═1d40ddef-d62e-4edb-b041-521c5c07fd05
# ╟─c2bf4eab-d078-4cf5-b309-bade8ea84864
# ╠═36e9f710-310f-41be-8de3-98d406f6e718
# ╠═91e6b75a-f8da-4a25-8886-898ec1678b75
# ╠═6f2c0b9f-83fe-484a-8176-6bc0847d9ad4
# ╠═d8ec968d-f3d5-47c0-bb6d-b07d56e2c0ce
# ╟─a3bed194-ea59-4b7e-aa99-0eea6f879c10
# ╠═a7243d4d-c706-412f-9a83-1324b6742c40
# ╠═2577fa49-97bc-43d7-a60b-3885474050e7
# ╠═634c40b5-299f-4b77-8e07-2c5a15a73e06
# ╠═c5be6137-f1e5-4dba-be37-ddbe23c3fd1d
# ╠═6ee68a92-ad08-46a7-bde0-c66b131114b3
# ╟─2efdbe51-0ab9-474f-a749-d04c7341feb7
# ╠═df689303-2d27-4d68-b4a5-05491360f7f0
# ╠═18919f7d-b077-465e-a360-c99de01821b3
# ╠═7c94084c-4519-4e2a-927e-19160726203d
# ╠═3c1ed4ad-f4ab-4052-974b-f41953f4300a
# ╟─97e33422-6d1b-4bc4-a86b-e50536d5537a
# ╟─490b7c22-1a07-4506-b4bb-1bdb6e191ca1
# ╠═594e804a-9589-4787-8612-1c51b1d324c2
# ╠═8ac3435d-1375-4e0e-b2dd-d48388542e3b
# ╠═5ac94c34-d0f7-4b4d-b33c-3f513d7a510d
# ╠═055ca4a5-a451-4cd5-ae36-2879a3d0fded
# ╟─0300b3df-e65c-43e6-83bb-01ef7235f76e
# ╠═01754ba7-edda-4304-b98a-040ded8231dc
# ╠═ac5ae896-bc8e-414a-b47f-924fa201be91
# ╠═4d5efea3-6363-4144-9cc3-7225d94a9011
# ╠═723a0f8f-82ec-4f0a-8c2a-58c43bf1cfb7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
