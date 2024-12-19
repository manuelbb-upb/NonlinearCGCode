### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 2cc74d18-b87d-11ef-0afe-077b7f4108d3
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(@__DIR__)
	Pkg.develop(; url=joinpath(@__DIR__, "NonlinearCG"))
	Pkg.develop(; url=joinpath(@__DIR__, "TestProblems"))
	Pkg.add("PlutoHooks")
	Pkg.add("PlutoLinks")
	Pkg.add("PlutoUI")
	Pkg.add("DataFrames")
	Pkg.add("DataStructures")
	Pkg.add("LaTeXTabulars")
	Pkg.add("LaTeXStrings")
	Pkg.add("ProgressLogging")
	Pkg.add("JLD2")
end

# ╔═╡ 61eae34f-e8d3-4afb-ac6f-f6b33096373f
begin
	using PlutoUI: TableOfContents
	TableOfContents()
end

# ╔═╡ c5f5b5ff-443c-4e53-9a32-48235b27cc1e
begin
	using PlutoHooks: @use_memo
	using PlutoUI: CounterButton
end

# ╔═╡ 52d795cf-1963-4f6b-895c-0d663fab2451
using LaTeXTabulars

# ╔═╡ 11a1e178-c960-46f1-9cfd-b290eb4f058b
using LaTeXStrings

# ╔═╡ 92c3fcc1-79b1-48d6-8f4f-f087562feac9
using Printf

# ╔═╡ 8564d05c-1c37-4bb8-bbac-9cecf3781e11
using ProgressLogging

# ╔═╡ e5c0f5ce-5fe6-418f-807d-b701a92f2427
using JLD2

# ╔═╡ 149ef455-62bb-47e8-a77d-6a3c5eaa8444
md"## Dependencies"

# ╔═╡ cb0f7e62-2f83-4487-b5b8-7a9ee2f7df87
# ╠═╡ show_logs = false
import PlutoLinks: @revise

# ╔═╡ 0735af1b-d5a8-4906-92be-5edbffe58465
@revise using NonlinearCG

# ╔═╡ e438b191-0771-4bf1-840b-6f6aac9a49e7
begin 
	@revise import TestProblems
	TP = TestProblems
end

# ╔═╡ 5cdf5346-f2b6-483d-80d4-1a320b7153a7
import DataStructures: OrderedDict

# ╔═╡ 06269d0e-dfcf-4dd6-96b1-a5c7a26e3993
import DataFrames as DF

# ╔═╡ 7bd1b6ba-cebf-4c61-943d-7a513ddbd08e
import Logging: Debug, Info

# ╔═╡ 427ee390-2ae4-4fed-a801-bcc5e58bc17c
import Random

# ╔═╡ 186077e1-3e33-43ab-8dd7-0e2f242a836c
md"## Setup"

# ╔═╡ 87831221-4496-4f76-866e-7e4e610e1c30
md"""
The test problems are defined in `TestProblems`. 
Each test problem has its own data type that specifies objective and gradient 
evaluation via dispatch.
We build a dictionary of all test problems, with test problem objects already 
instantiated.
They are later converted to a `MOP` that is consumed by `optimize`.
"""

# ╔═╡ 45d78cff-a07e-4f79-96e0-af05e07a0ae9
PROBLEMS = OrderedDict{String, Any}(
	"BK1" => TP.BK1(),
	"DD1a" => TP.DDS1a(),
	#"DDS1b" => TP.DDS1b(),
	"DD1b" => TP.DDS1c(), 	# DD1b in chenConjugateGradientMethods2024
	"DD1c" => TP.DDS1d(),  # DD1c in chenConjugateGradientMethods2024
	"DGO1" => TP.DGO1(),
	"Far1" => TP.Far1(),
	"FDSa" => TP.FDS(; num_vars = 10),
	"FDSb" => TP.FDS(; num_vars = 200),
	"FDSc" => TP.FDS(; num_vars = 500),
	"FDSd" => TP.FDS(; num_vars = 1000),
	"FF1" => TP.FF1(),
	"Hil1" => TP.Hil1(),
	"IKK1" => TP.IKK1(),
	#"IM1" => TP.IM1(), 	# domain error because of `sqrt`
	"JOS1a" => TP.JOS1a(),
	"JOS1b" => TP.JOS1b(),
	"JOS1c" => TP.JOS1c(),
	"KW2" => TP.KW2(),
	"Lov1" => TP.Lov1(),
	"Lov3" => TP.Lov3(),
	"Lov4" => TP.Lov4(),
	"Lov5" => TP.Lov5(),
	"MGH16" => TP.MGH16(),
	"MGH26" => TP.MGH26(),
	"MMR5a" => TP.MMR5a(),
	"MMR5b" => TP.MMR5b(),
	"MMR5c" => TP.MMR5c(),
	"MOP2" => TP.MOP2(; L=-2, U=2), 	# [-4, 4] in chenConjugateGradientMethods2024
										# but gradients too flat near bounds
	"MOP3" => TP.MOP3(),
	"MOP5" => TP.MOP5(),
	"PNR" => TP.PNR(),
	"SLCDT2" => TP.SLCDT2(),
	"SP1" => TP.SP1(),
	"SSFYY2" => TP.SSFYY2(),
	"TOI9" => TP.TOI9(),
	"TOI10" => TP.TOI10(),
	"VU1" => TP.VU1(),
	"RB2D" => TP.RB2D(),
)

# ╔═╡ 14b91d39-f604-47e7-8265-474a8a5ca53a
md"""
The function `optimize` dispatches on configuration objects of type `AbstractStepRule`.
Such objects define the descent algorithm flavor.
We have a function to return a dictionary of all relevant step rules based on the backtracking configuration:
"""

# ╔═╡ 7f67656b-f987-40b1-ba16-f67cd29b0298
function make_step_rule_dict(;
	armijo_mode=Val(:all),
	armijo_constant=1e-4,
	armijo_factor=0.5
)
	sz_rule_func(is_modified) = ArmijoBacktracking(;
		is_modified,
		mode = armijo_mode,
		factor = armijo_factor,
		constant = armijo_constant,
	)
	sz_rule_sd = sz_rule_func(Val(false))
	sz_rule = sz_rule_func(Val(true))
	
	step_rules = OrderedDict{String, Any}(
		"SD" => SteepestDescentDirection(;sz_rule = sz_rule_sd),
		"PRP3" => PRP3(;sz_rule),
		"PRPP" => PRPConeProjection(;sz_rule),		
		"FRR" => FletcherReevesRestart(;sz_rule),
		"FRF1a" => FletcherReevesFractionalLP1(;sz_rule),
		"FRF1b" => FletcherReevesFractionalLP1(;sz_rule, constant=10),
		"FRF2a" => FletcherReevesFractionalLP2(;sz_rule),
		"FRF2b" => FletcherReevesFractionalLP2(;sz_rule, constant=10),
		"FRBOa" => FletcherReevesBalancingOffset(;sz_rule),
		"FRBOb" => FletcherReevesBalancingOffset(;sz_rule, c_gamma=0.1)
	)
	return step_rules
end

# ╔═╡ 9d55cf0d-4bab-4f1f-92d9-d27f4d1e46a0
md"
The `optimize` function is wrapped by `run_mop`.
In `run_mop`, we first take an `AbstractTestProblem` and make a `MOP` out of it.
We then call `optimize` with the specified settings:
"

# ╔═╡ 97884128-1af8-4a05-a561-8cddc45a70cd
function run_mop(
	tp, x0, step_rule;
	log_level = Debug,
	crit_tol_abs = 1e-4,
	max_iter = 1000,
	x_tol_rel = 0,
	x_tol_abs = 0,
	fx_tol_rel = 0,
	fx_tol_abs = 0,
)
	mop = MOP(;
		dim_in = TP.num_variables(tp),
		dim_out = TP.num_objectives(tp),
		objectives! = TP.IPObjectivesClosure(tp),
		jac! = TP.IPJacClosure(tp)
	)
	res_tuple = optimize(
		x0, mop;
		step_rule,
		crit_tol_abs,
		log_level,
		max_iter,
		x_tol_rel,
		x_tol_abs,
		fx_tol_rel,
		fx_tol_abs
	)
	return res_tuple
end

# ╔═╡ 8b1599c8-f952-4d48-9763-2fc231d91cee
md"### Helpers"

# ╔═╡ 19e0c1cd-5fca-4f3b-bd19-139bbdcba05f
md"Initialize an empty result dataframe, columns indexed by step rule labels:"

# ╔═╡ 81c8bd42-831c-4d11-a742-27656176a9e5
function result_dataframe(step_rule_names)
	return DF.DataFrame(
		(cname => Float64[] for cname in step_rule_names)...
	)
end

# ╔═╡ 993d13de-fc12-4a50-9e08-c39e135a7648
md"Parse the backtracking mode (value type) as a string for filenames etc.:"

# ╔═╡ 8653f957-0389-45f1-a8bf-fe8f175afc0c
modestring(::Val{F})where F="$(string(F))"

# ╔═╡ ef16b140-9921-466a-8eed-6087a1c413e4
md"## Run"

# ╔═╡ d453e7aa-9bba-4681-872b-b73b8c6c79fa
md"### Run Button"

# ╔═╡ 4cb1d9b6-76e6-46e8-a918-8ae8e02ea1db
@bind run1 CounterButton("Run Experiments")

# ╔═╡ 9b68cb0b-f2d1-4872-b186-dc0304cf9a40
md"## LaTeX Utils"

# ╔═╡ d4c3eb7e-fa4e-4672-ac7e-47cde0ba6aeb
"""
Take a result dataframe and format its entries.
Entries will be `String` or `LaTeXString`."""
function latexify_df(
	df;
	comp = :min_best,
	skipcols = [],
)
	ldf = DF.DataFrame(
		(cname => LaTeXString[] for cname in names(df))...
	)
	for row = eachrow(df)
		_row = row[2:end]
		nums = []
		for (cname, entry) in pairs(row)
			!(entry isa Number) && continue
			(cname in skipcols) && continue
			push!(nums, entry)
		end
		unique!(nums)

		rmin, rmax = extrema(nums)
		rw = rmax - rmin
		
		if comp == :min_best
			if iszero(rw)
				colorfunc = n -> 20
			else
				# smallest value should be darkest, have value 50
				# biggest value should be lightest, have value 20
				colorfunc = n -> floor(Int, 20 + 30 * ((n - rmin) / rw))
			end
		else
			if iszero(rw)
				colorfunc = n -> 20
			else
				colorfunc = n -> floor(Int, 50 - 30 * ((n - rmin) / rw))
			end
		end
		new_row = Dict()
		for i=eachindex(row)
			j=row[i]
			if ismissing(j)
				new_row[i] = "NaN"
				continue
			end
			if j isa AbstractString
				new_row[i] = j
				continue
			end
			_j = @sprintf("%g", j)
			if i in skipcols
				#new_row[i] = L"%$(_j)"
				new_row[i] = LaTeXString("\\cellcolor{black!55}\$$(_j)\$")
				continue
			end
			new_row[i] = LaTeXString("\\cellcolor{black!$(colorfunc(j))}\$$(_j)\$") #L"\mathbf{%$_j}"
		end
		push!(ldf, new_row)
	end

	return ldf
end

# ╔═╡ 076c32ae-8241-4ff4-b640-230629003f40
"""
Take a preprocessed result dataframe and return `tex` code suitable for 
compilation in a LaTeX document."""
function latex_code(
	df;
	io=String,
	title=nothing,
)
	nrows, ncols = size(df)
	
	colspec = "l|" * join("c" for _ in 2:ncols)

	cols = collect(names(df))
	rows = Any[
		cols,
		Rule(:mid),
		collect(eachrow(df))...,
		Rule(:bottom)
	]

	if !isnothing(title)
		pushfirst!(rows, Rule(:mid))
		pushfirst!(rows, [MultiColumn(ncols, :c, title)])
	end

	pushfirst!(rows, Rule(:top))
	
	latex_tabular(io, Tabular(colspec), rows)
end

# ╔═╡ 2b1b8535-6470-4b72-ac36-cb516a131ba8
function make_latex(
	df;
	resname = "its",
	resword = "Number of Iterations",
	title = nothing,
	armijo_mode, 
	fname_prefix, 
	crit_tol_abs,
	comp = :min_best,
	skipcols = []
)
	ldf = latexify_df(df; comp, skipcols)

	modestr = modestring(armijo_mode)
	fname = "$(fname_prefix)_num_$(resname)_$(modestr)_$(@sprintf("%.e", crit_tol_abs)).tex"

	if isnothing(title)
		title = "$(resword), mode=$(modestr), tol=$(@sprintf("%.e", crit_tol_abs))"
	end

	return latex_code(ldf; io = fname, title)
end

# ╔═╡ a18e5d2d-0b5a-4c6d-bbd6-eac4661c36d6
function result_latex_tables(
	df_its, df_fcalls, df_gcalls, df_solved;
	armijo_mode, fname_prefix, crit_tol_abs,
)
	make_latex(
		df_its;
		armijo_mode, fname_prefix, crit_tol_abs,
		resname = "its", resword = "Number of Iterations",
		skipcols = [:max,]
	)
	make_latex(
		df_fcalls;
		armijo_mode, fname_prefix, crit_tol_abs,
		resname = "fcalls", resword = "Number of Func. Calls",
	)
	make_latex(
		df_gcalls;
		armijo_mode, fname_prefix, crit_tol_abs,
		resname = "gcalls", resword = "Number of Grad. Calls",
	)
	make_latex(
		df_solved;
		armijo_mode, fname_prefix, crit_tol_abs,
		resname = "solved", resword = "Solved Percentage",
		comp = :max_best
	)
	return nothing
end

# ╔═╡ b8a46fb4-c976-4681-81e5-daa98cedf53b
function run_experiments(;
	num_x0 = 10,
	armijo_mode=Val(:all),
	armijo_constant=1e-4,
	armijo_factor=0.5,
	max_iter=1000,
	crit_tol_abs=1e-6,
	x_tol_rel=1e-15,
	log_level=Debug,
	scale_problems=true,
	fname_prefix="",
	make_latex_tables=false,
	save_dataframes=true,
	overwrite=true
)
	global PROBLEMS

	if max_iter isa Number
		max_iter_func = n -> max_iter
	else
		max_iter_func = max_iter
	end
	
	Random.seed!(1618)

	res_fpath = "dataframes_$(modestring(armijo_mode))_$(@sprintf("%.e", crit_tol_abs)).jld2"

	if overwrite || !isfile(res_fpath)

		STEP_RULES = make_step_rule_dict(; armijo_mode, armijo_constant, armijo_factor)
	
		step_rule_names = collect(keys(STEP_RULES))
		df_solved = result_dataframe(step_rule_names)
		df_its = result_dataframe(["max"; step_rule_names])
		df_fcalls = result_dataframe(step_rule_names)
		df_gcalls = result_dataframe(step_rule_names)
	
		agg_subdf(sdf) = first(DF.combine(sdf, DF.Not(:x0) .=> DF.median; renamecols=false))
	
		add_pnames!(df, problem_names) = begin
			df.Problem .= problem_names
			DF.select!(df, "Problem", DF.Not("Problem"))
		end
	
		problem_names = String[]
		
		all_rules = collect(pairs(STEP_RULES))	# collect for Threads
		tlock = ReentrantLock()					# lock for parallel execution
		#@progress for (tp_name, tp) in PROBLEMS
		@progress for tp_name in collect(keys(PROBLEMS))
			tp = PROBLEMS[tp_name]
			@info "Problem $(tp_name)"
			push!(problem_names, tp_name)
			
			num_vars = TP.num_variables(tp)
			lb = TP.lower_variable_bounds(tp)
			ub = TP.upper_variable_bounds(tp)
	
			wb = ub .- lb
			X0 = lb .+ wb .* rand(num_vars, num_x0)
			
			X0s = copy.(eachcol(X0))
			
			d_its = DF.DataFrame( :x0 => X0s )
			d_fcalls = DF.DataFrame( :x0 => X0s )
			d_gcalls = DF.DataFrame( :x0 => X0s )
	
			d_solved = Dict{String, Float64}()
	
			max_iter_tp = max_iter_func(TP.num_variables(tp))
			
			Threads.@threads for (sr_name, step_rule) in all_rules
				@info "\t * $(sr_name)"
				#!(sr_name == "PRP3") && continue
	
				solved = 0
				its = Int[]
				fcalls = Int[]
				gcalls = Int[]
				
				for x0 in eachcol(X0)
					#@info "$(NonlinearCG.pretty_row_vec(x0))"
					rtp = scale_problems ? TP.make_scaled(tp, x0) : tp
	
					res_tuple = run_mop(
						rtp, x0, step_rule; 
						max_iter = max_iter_tp,
						crit_tol_abs, log_level, x_tol_rel,
					)
	
					if res_tuple.critval < crit_tol_abs
						solved += 1
					end
	
					n_its = res_tuple.num_its
					n_fcalls = res_tuple.num_func_calls
					n_gcalls = res_tuple.num_grad_calls
					#@info "n_its = $(n_its), n_fcalls = $(n_fcalls), n_gcalls = $(n_gcalls)."
					push!(its, n_its)
					push!(fcalls, n_fcalls)
					push!(gcalls, n_gcalls)
				end
	
				lock(tlock) do
					d_its[:, sr_name] .= copy(its)
					d_fcalls[:, sr_name] .= copy(fcalls)
					d_gcalls[:, sr_name] .= copy(gcalls)
					d_solved[sr_name] = solved / num_x0 * 100
				end
			end
			
			push!(df_its, agg_subdf(d_its); cols=:subset)
			df_its[end, :max] = max_iter_tp
			push!(df_fcalls, agg_subdf(d_fcalls); cols=:subset)
			push!(df_gcalls, agg_subdf(d_gcalls); cols=:subset)
			push!(df_solved, d_solved; cols=:subset)
	
			#length(problem_names) >= 3 && break
		end
		add_pnames!(df_its, problem_names)
		add_pnames!(df_fcalls, problem_names)
		add_pnames!(df_gcalls, problem_names)
		add_pnames!(df_solved, problem_names)
	
		if save_dataframes
			jldsave(
				res_fpath;
				df_its, df_fcalls, df_gcalls, df_solved
			)
		end
	else
		df_its, df_fcalls, df_gcalls, df_solved = load(res_fpath, "df_its", "df_fcalls", "df_gcalls", "df_solved")
	end

	if make_latex_tables
		@info "Loading $(res_fpath)."
		result_latex_tables(
			df_its, df_fcalls, df_gcalls, df_solved;
			armijo_mode, fname_prefix, crit_tol_abs
		)
	end
	
	df_its, df_solved, df_fcalls, df_gcalls
end

# ╔═╡ 6ee32cf1-1a5c-459a-8be7-a4c428e8cf18
(
	df_its_1e_6, df_fcalls_1e_6, df_gcalls_1e_6, df_solved_1e_6
) = @use_memo([run1]) do
	if run1 >= 1
		run_experiments(; 
			scale_problems=true, 
			num_x0=100, 
			crit_tol_abs=1e-6, 
			x_tol_rel=1e-10,
			max_iter= n -> max(10 * n, 1000),
			armijo_mode=Val(:all),
			make_latex_tables = true,
			overwrite = false			
		)
	end
end

# ╔═╡ 74ffcd43-d36b-4ad5-8609-232b404d288e
md"### Problem Table"

# ╔═╡ b0783a63-5064-43e0-bd89-b04804260951
begin
	pdf = DF.DataFrame(
		name = String[],
		num_vars = Int[],
		num_objectives = Int[],
		lb = Vector{Float64}[],
		ub = Vector{Float64}[],
		refs = Vector{String}[]
	)
	for (name, tp) in pairs(PROBLEMS)
		num_vars = TP.num_variables(tp)
		num_objectives = TP.num_objectives(tp)
		lb = TP.lower_variable_bounds(tp)
		ub = TP.upper_variable_bounds(tp)
		refs = let rk=TP.reference_keys(tp); isnothing(rk) ? [] : collect(rk) end
		push!(pdf, (; name, num_vars, num_objectives, lb, ub, refs))
	end

	pdf
end

# ╔═╡ aa78709d-3d20-4462-834d-7830d318de3c
function bound_strings(bounds)
	b_strings = []
	for bvec in bounds
		b = first(bvec)
		if all(isapprox(b), bvec)
			_b = isapprox(b, π) ? "\\pi" : isapprox(b, -π) ? "-\\pi" : @sprintf("%g", b)
			_s = "\$[" * _b * ", \\ldots, " * _b * "]\$"
		else
			_s = "\$[" * join([@sprintf("%g", _b) for _b in bvec], ", ") * "]\$"
		end
		s = LaTeXString(_s)
		push!(b_strings, s)
	end
	return b_strings
end	

# ╔═╡ b2d13f8d-a9d9-4a05-9041-085703b1cbb2
function ref_strings(refs)
	r_strings = []
	for rvec in refs
		if isempty(rvec)
			push!(r_strings, "")
			continue
		end
		s = LaTeXString("\\cite{" * join(rvec, ",") * "}")
		push!(r_strings, s)
	end
	return r_strings
end

# ╔═╡ 790ffbcb-0cfe-467d-b515-8285e79858ae
begin
	lpdf = DF.select(pdf, [:name, :num_vars, :num_objectives])
	lpdf.lb = bound_strings(pdf.lb)
	lpdf.ub = bound_strings(pdf.ub)
	lpdf.refs = ref_strings(pdf.refs)

	lpdf
end

# ╔═╡ 5478dfa9-55d2-4d72-9d34-c60e02952b34
let
	nrows, ncols = size(lpdf)
	
	colspec = "l|" * join("c" for _ in 2:ncols)

	cols = [
		"Name",
		L"\dimIn",
		L"\dimOut",
		"lower bounds",
		"upper bounds",
		"Ref."
	]
	rows = Any[
		cols,
		Rule(:mid),
		collect(eachrow(lpdf))...,
		Rule(:bottom)
	]

	pushfirst!(rows, Rule(:top))
	
	latex_tabular("problem_table.tex", Tabular(colspec), rows)
end

# ╔═╡ 74de0d46-dabb-41df-a45d-af5d84dc3a77
md"## Temporary"

# ╔═╡ bf1fa747-d17e-42f5-b9d3-5dddf535214b
# ╠═╡ disabled = true
#=╠═╡
let tp = TP.MMR5c();
	#tp = TP.DGO1()

	step_rule = PRPConeProjection(;
		sz_rule = ArmijoBacktracking(;
			mode=Val(:all),
			constant=1e-4,
		)
	);
	step_rule = FletcherReevesFractionalLP1(;
		constant = 10,
		sz_rule = ArmijoBacktracking(;
			mode=Val(:all),
			constant=1e-15,
		)
	);
	#step_rule = SteepestDescentDirection()
	n = TP.num_variables(tp)
	
	lb = TP.lower_variable_bounds(tp)
	ub = TP.upper_variable_bounds(tp)

	x0 = lb .+ (ub .- lb) .* rand(n)
	
	#rtp = TP.make_scaled(tp, x0)
	rtp = tp
	res = run_mop(
		rtp, x0, step_rule;
		max_iter=max(10 * n, 1000),
		crit_tol_abs=1e-6,
		x_tol_rel=1e-10,
	)
	display(x0)
	res
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═61eae34f-e8d3-4afb-ac6f-f6b33096373f
# ╟─149ef455-62bb-47e8-a77d-6a3c5eaa8444
# ╠═2cc74d18-b87d-11ef-0afe-077b7f4108d3
# ╠═c5f5b5ff-443c-4e53-9a32-48235b27cc1e
# ╠═cb0f7e62-2f83-4487-b5b8-7a9ee2f7df87
# ╠═0735af1b-d5a8-4906-92be-5edbffe58465
# ╠═e438b191-0771-4bf1-840b-6f6aac9a49e7
# ╠═5cdf5346-f2b6-483d-80d4-1a320b7153a7
# ╠═06269d0e-dfcf-4dd6-96b1-a5c7a26e3993
# ╠═52d795cf-1963-4f6b-895c-0d663fab2451
# ╠═11a1e178-c960-46f1-9cfd-b290eb4f058b
# ╠═7bd1b6ba-cebf-4c61-943d-7a513ddbd08e
# ╠═427ee390-2ae4-4fed-a801-bcc5e58bc17c
# ╠═92c3fcc1-79b1-48d6-8f4f-f087562feac9
# ╠═8564d05c-1c37-4bb8-bbac-9cecf3781e11
# ╠═e5c0f5ce-5fe6-418f-807d-b701a92f2427
# ╟─186077e1-3e33-43ab-8dd7-0e2f242a836c
# ╟─87831221-4496-4f76-866e-7e4e610e1c30
# ╠═45d78cff-a07e-4f79-96e0-af05e07a0ae9
# ╟─14b91d39-f604-47e7-8265-474a8a5ca53a
# ╠═7f67656b-f987-40b1-ba16-f67cd29b0298
# ╟─9d55cf0d-4bab-4f1f-92d9-d27f4d1e46a0
# ╠═97884128-1af8-4a05-a561-8cddc45a70cd
# ╟─8b1599c8-f952-4d48-9763-2fc231d91cee
# ╟─19e0c1cd-5fca-4f3b-bd19-139bbdcba05f
# ╠═81c8bd42-831c-4d11-a742-27656176a9e5
# ╟─993d13de-fc12-4a50-9e08-c39e135a7648
# ╠═8653f957-0389-45f1-a8bf-fe8f175afc0c
# ╟─ef16b140-9921-466a-8eed-6087a1c413e4
# ╠═b8a46fb4-c976-4681-81e5-daa98cedf53b
# ╠═a18e5d2d-0b5a-4c6d-bbd6-eac4661c36d6
# ╠═2b1b8535-6470-4b72-ac36-cb516a131ba8
# ╟─d453e7aa-9bba-4681-872b-b73b8c6c79fa
# ╠═4cb1d9b6-76e6-46e8-a918-8ae8e02ea1db
# ╠═6ee32cf1-1a5c-459a-8be7-a4c428e8cf18
# ╟─9b68cb0b-f2d1-4872-b186-dc0304cf9a40
# ╠═d4c3eb7e-fa4e-4672-ac7e-47cde0ba6aeb
# ╠═076c32ae-8241-4ff4-b640-230629003f40
# ╟─74ffcd43-d36b-4ad5-8609-232b404d288e
# ╠═b0783a63-5064-43e0-bd89-b04804260951
# ╠═aa78709d-3d20-4462-834d-7830d318de3c
# ╠═b2d13f8d-a9d9-4a05-9041-085703b1cbb2
# ╠═790ffbcb-0cfe-467d-b515-8285e79858ae
# ╠═5478dfa9-55d2-4d72-9d34-c60e02952b34
# ╟─74de0d46-dabb-41df-a45d-af5d84dc3a77
# ╠═bf1fa747-d17e-42f5-b9d3-5dddf535214b
