### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 2cc74d18-b87d-11ef-0afe-077b7f4108d3
begin
	using Pkg
	Pkg.activate(@__DIR__)
	Pkg.develop(; url=joinpath(@__DIR__, "NonlinearCG"))
	Pkg.develop(; url=joinpath(@__DIR__, "TestProblems"))
	Pkg.add("PlutoLinks")
	Pkg.add("DataFrames")
	Pkg.add("DataStructures")
	Pkg.add("LaTeXTabulars")
	Pkg.add("LaTeXStrings")
	Pkg.add("ProgressLogging")
end

# ╔═╡ 52d795cf-1963-4f6b-895c-0d663fab2451
using LaTeXTabulars

# ╔═╡ 11a1e178-c960-46f1-9cfd-b290eb4f058b
using LaTeXStrings

# ╔═╡ 92c3fcc1-79b1-48d6-8f4f-f087562feac9
using Printf

# ╔═╡ 8564d05c-1c37-4bb8-bbac-9cecf3781e11
using ProgressLogging

# ╔═╡ cb0f7e62-2f83-4487-b5b8-7a9ee2f7df87
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

# ╔═╡ 45d78cff-a07e-4f79-96e0-af05e07a0ae9
PROBLEMS = OrderedDict{String, Any}(
	"BK1" => TP.BK1(),
	"DDS1a" => TP.DDS1a(),
	"DDS1b" => TP.DDS1b(),
	"DDS1c" => TP.DDS1a(),
	"DGO1" => TP.DGO1(),
	"RB2D" => TP.RB2D()
)

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
		"FRF1_1" => FletcherReevesFractionalLP1(;sz_rule),
		"FRF1_10" => FletcherReevesFractionalLP1(;sz_rule, constant=10),
		"FRF2_1" => FletcherReevesFractionalLP2(;sz_rule),
		"FRF2_10" => FletcherReevesFractionalLP2(;sz_rule, constant=10),
		"FRBO" => FletcherReevesBalancingOffset(;sz_rule),
		"FRBO_01" => FletcherReevesBalancingOffset(;sz_rule, c_gamma=0.1)
	)
	return step_rules
end

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

# ╔═╡ 81c8bd42-831c-4d11-a742-27656176a9e5
function result_dataframe(step_rule_dict)
	return DF.DataFrame(
		(cname => Float64[] for cname in keys(step_rule_dict))...
	)
end

# ╔═╡ 8653f957-0389-45f1-a8bf-fe8f175afc0c
modestring(::Val{F})where F="$(string(F))"

# ╔═╡ d4c3eb7e-fa4e-4672-ac7e-47cde0ba6aeb
function latexify_df(df)
	ldf = DF.DataFrame(
		(cname => LaTeXString[] for cname in names(df))...
	)
	for row = eachrow(df)
		_row = row[2:end]
		k, t = first(pairs(_row))
		for (_k, _t) in pairs(_row)
			ismissing(_t) && continue
			if ismissing(t) || _t <= t
				k = _k
				t = _t
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
			if j==t
				new_row[i] = L"\mathbf{%$_j}"
			else
				new_row[i] = L"%$_j"
			end
		end
		push!(ldf, new_row)
	end

	return ldf
end

# ╔═╡ 076c32ae-8241-4ff4-b640-230629003f40
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

# ╔═╡ b8a46fb4-c976-4681-81e5-daa98cedf53b
function run_experiments(;
	num_x0 = 10,
	armijo_mode=Val(:all),
	armijo_constant=1e-4,
	armijo_factor=0.5,
	crit_tol_abs=1e-6,
	log_level=Debug,
	scale_problems=true,
	fname_prefix=""
)
	@show num_x0
	@show crit_tol_abs
	Random.seed!(1618)

	STEP_RULES = make_step_rule_dict(; armijo_mode, armijo_constant, armijo_factor)
	
	df_its = result_dataframe(STEP_RULES)
	df_fcalls = result_dataframe(STEP_RULES)
	df_gcalls = result_dataframe(STEP_RULES)

	agg_subdf(sdf) = first(DF.combine(sdf, DF.Not(:x0) .=> DF.median; renamecols=false))

	add_pnames!(df, problem_names) = begin
		df.Problem .= problem_names
		DF.select!(df, "Problem", DF.Not("Problem"))
	end

	problem_names = String[]
	#@progress for (tp_name, tp) in PROBLEMS
	@progress for tp_name in collect(keys(PROBLEMS))
		tp = PROBLEMS[tp_name]
		#!(tp_name == "DDS1a") && continue
		
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
		for (sr_name, step_rule) in pairs(STEP_RULES)
			#!(sr_name == "PRP3") && continue
			
			its = Int[]
			fcalls = Int[]
			gcalls = Int[]
			
			for x0 in eachcol(X0)
				#@info "$(NonlinearCG.pretty_row_vec(x0))"
				rtp = scale_problems ? TP.make_scaled(tp, x0) : tp

				res_tuple = run_mop(rtp, x0, step_rule; crit_tol_abs, log_level)

				n_its = res_tuple.num_its
				n_fcalls = res_tuple.num_func_calls
				n_gcalls = res_tuple.num_grad_calls
				#@info "n_its = $(n_its), n_fcalls = $(n_fcalls), n_gcalls = $(n_gcalls)."
				push!(its, n_its)
				push!(fcalls, n_fcalls)
				push!(gcalls, n_gcalls)
			end
			d_its[:, sr_name] .= copy(its)
			d_fcalls[:, sr_name] .= copy(fcalls)
			d_gcalls[:, sr_name] .= copy(gcalls)
		end
		push!(df_its, agg_subdf(d_its); cols=:subset)
		push!(df_fcalls, agg_subdf(d_fcalls); cols=:subset)
		push!(df_gcalls, agg_subdf(d_gcalls); cols=:subset)
	end
	add_pnames!(df_its, problem_names)
	add_pnames!(df_fcalls, problem_names)
	add_pnames!(df_gcalls, problem_names)

	ldf_its = latexify_df(df_its)
	ldf_fcalls = latexify_df(df_fcalls)
	ldf_gcalls = latexify_df(df_gcalls)
	
	fname_its = "$(fname_prefix)_num_its_$(modestring(armijo_mode))_$(@sprintf("%.e", crit_tol_abs)).tex"
	fname_fcalls = "$(fname_prefix)_num_fcalls_$(modestring(armijo_mode))_$(@sprintf("%.e", crit_tol_abs)).tex"
	fname_gcalls = "$(fname_prefix)_num_gcalls_$(modestring(armijo_mode))_$(@sprintf("%.e", crit_tol_abs)).tex"

	title_its = "Number of Iterations, mode=$(modestring(armijo_mode)), tol=$(@sprintf("%.e", crit_tol_abs))"
	title_fcalls = "Number of Func. Calls, mode=$(modestring(armijo_mode)), tol=$(@sprintf("%.e", crit_tol_abs))"
	title_gcalls = "Number of Grad. Calls, mode=$(modestring(armijo_mode)), tol=$(@sprintf("%.e", crit_tol_abs))"

	latex_code(ldf_its; io=fname_its, title=title_its)
	latex_code(ldf_fcalls; io=fname_fcalls, title=title_fcalls)
	latex_code(ldf_gcalls; io=fname_gcalls, title=title_gcalls)
	
	df_its, df_fcalls, df_gcalls
end

# ╔═╡ bf1fa747-d17e-42f5-b9d3-5dddf535214b
let 
	x0 = [-4.56e+00, 7.46e-01]
	tp = TP.BK1()
	tp = TP.DDS1a()
	tp = TP.RB2D()
	rtp = TP.make_scaled(tp, x0)
	rtp = rtp
	run_mop(
		rtp, x0, PRP3(;
			sz_rule = ArmijoBacktracking(;
				mode=Val(:all),
				constant=1e-4,
			)
		);
		max_iter=50,
		crit_tol_abs=1e-10,
		log_level=Info
	)
end

# ╔═╡ 6ee32cf1-1a5c-459a-8be7-a4c428e8cf18
run_experiments(; scale_problems=false, num_x0=10, crit_tol_abs=1e-6, armijo_mode=Val(:all))

# ╔═╡ Cell order:
# ╠═2cc74d18-b87d-11ef-0afe-077b7f4108d3
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
# ╠═45d78cff-a07e-4f79-96e0-af05e07a0ae9
# ╠═7f67656b-f987-40b1-ba16-f67cd29b0298
# ╠═97884128-1af8-4a05-a561-8cddc45a70cd
# ╠═81c8bd42-831c-4d11-a742-27656176a9e5
# ╠═8653f957-0389-45f1-a8bf-fe8f175afc0c
# ╠═b8a46fb4-c976-4681-81e5-daa98cedf53b
# ╠═d4c3eb7e-fa4e-4672-ac7e-47cde0ba6aeb
# ╠═076c32ae-8241-4ff4-b640-230629003f40
# ╠═bf1fa747-d17e-42f5-b9d3-5dddf535214b
# ╠═6ee32cf1-1a5c-459a-8be7-a4c428e8cf18
