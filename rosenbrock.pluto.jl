### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ a5953b72-bd63-11ef-1190-ffea7b2dca65
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(@__DIR__)
	Pkg.develop(; url=joinpath(@__DIR__, "NonlinearCG"))
	Pkg.develop(; url=joinpath(@__DIR__, "TestProblems"))
	Pkg.add("LaTeXStrings")
	Pkg.add("CairoMakie")
	Pkg.add("PlutoLinks")
	Pkg.add("HaltonSequences")
	#Pkg.add("ColorSchemes")
end

# ╔═╡ f56ea3de-cae3-43a0-8c59-33d695b6864f
using PlutoLinks: @revise

# ╔═╡ 5c8fe52a-84d6-49af-b859-0c9a25354d71
using CairoMakie

# ╔═╡ 0e732d21-b4a6-4dcd-ba46-c7e50c67f552
using LaTeXStrings

# ╔═╡ 4c788894-1a4b-4235-96a7-e3ba67f2cb77
using HaltonSequences: HaltonPoint

# ╔═╡ a4626e6d-8692-4695-ba1b-080bf4739fa3
using Printf

# ╔═╡ 52ecbc11-9345-4a66-bc3d-6374ae7ccf3c
begin
	@revise import TestProblems
	TP = TestProblems
end

# ╔═╡ d80c68e3-d3c5-4b00-9f68-3e97a5a55daf
begin
	@revise import NonlinearCG
	NCG = NonlinearCG
end

# ╔═╡ 561269bf-2e05-42cd-a0bf-613260da63cb
import Logging

# ╔═╡ f24ad06d-2498-4ab2-830d-465571f9d18a
begin
	rb2d = TP.RB2D()
	dim_in = TP.num_variables(rb2d)
	dim_out = TP.num_objectives(rb2d)
	lb = TP.lower_variable_bounds(rb2d)
	ub = TP.upper_variable_bounds(rb2d)
	wb = ub .- lb
	f = TP.OOPObjectivesClosure(rb2d)
	Df = TP.OOPJacClosure(rb2d)
	mop = NCG.MOP(;
		dim_in, dim_out,
		objectives! = TP.IPObjectivesClosure(rb2d),
		jac! = TP.IPJacClosure(rb2d)
	)
end

# ╔═╡ 243c9ca3-85d5-41ce-9388-03c376a74fb1
begin
struct CritMap{
	fw_cache_Type
}
	fw_cache :: fw_cache_Type
end

function CritMap(no::Integer)
	α = zeros(no)
	_α = similar(α)
	M = zeros(no, no)
	fw_cache = NCG.FrankWolfeCache(α, _α, M)
	return CritMap(fw_cache)
end

function (cm::CritMap)(x1, x2)
	J = Df([x1, x2])
	_, c = NCG.frank_wolfe_multidir_dual!(cm.fw_cache, J)
	return c
end
end

# ╔═╡ a1772153-f309-4feb-81af-565b9f4a7ad5
begin
	Hrun = lb .+ wb .* reduce(hcat, HaltonPoint(dim_in; length=8))
	#Hrun .*= (0.01 .* rand(2, 3))
end

# ╔═╡ 2749cf2a-6ebc-41d9-873b-95c7bd7c412a
step_rule_sd = NCG.SteepestDescentDirection()

# ╔═╡ e3e5fe5e-ff63-4afc-838e-450a093895e9
step_rule_prp = NCG.PRPConeProjection()

# ╔═╡ f917aeaf-e024-47ba-b405-5cdf64084c03
step_rule_fr = NCG.FletcherReevesBalancingOffset()

# ╔═╡ 5d88a5cc-0f41-4783-8173-3f3b47017eba
function make_runs(step_rule, X)
	xruns = Matrix{Float64}[]
	yruns = Matrix{Float64}[]
	for x0 = eachcol(X)
		callback = NCG.GatheringCallback(Float64)
		res = NCG.optimize(
			x0, mop;
			step_rule,
			callback,
			max_iter = 30,
			log_level = Logging.Debug
		)
		cback = res.callback
		xs = reduce(hcat, cback.x)
		ys = reduce(hcat, cback.fx)
		push!(xruns, xs)
		push!(yruns, ys)
	end
	return xruns, yruns
end

# ╔═╡ d9b49cbf-2595-4b98-b82b-c1047608f768
x_sd, y_sd = make_runs(step_rule_sd, Hrun)

# ╔═╡ cd5d5aa4-4757-4afc-8b45-8c6acbccb031
x_prp, y_prp = make_runs(step_rule_prp, Hrun)

# ╔═╡ ab4e7edc-7933-499e-a80d-80e102e514c4
x_fr, y_fr = make_runs(step_rule_fr, Hrun)

# ╔═╡ cbe727f9-f211-4996-99b0-f20099ba1261
c = CritMap(dim_out)

# ╔═╡ 57076e21-9ddf-43b0-9479-273e7f24ce50
_f(x1, x2) = f([x1; x2])

# ╔═╡ b59c9787-e1c2-4510-b069-ed8b411d3766
function mat_lims(mat)
	ex = extrema(mat; dims=2)
	lb = mapreduce(first, vcat, ex)
	ub = mapreduce(last, vcat, ex)
	return lb, ub
end

# ╔═╡ 6968952b-4807-4fdb-b46b-1c222081b499
function data_lims(Xs...)
	_lb = [Inf, Inf]
	_ub = [-Inf, -Inf]
	for X in Xs
		__lb, __ub = mat_lims(X)
		_lb = min.(_lb, __lb)
		_ub = max.(_ub, __ub)
	end
	return _lb, _ub
end

# ╔═╡ 1f0e1271-4535-40e2-972d-096c472d2b5c
begin
	ps1 = range(rb2d.a1, rb2d.a2, 50)
	ps2 = ps1.^2
	
	xps = mapreduce(((x1, x2),) -> [x1; x2], hcat, zip(ps1, ps2))
	fps = mapreduce(((x1, x2),) -> _f(x1, x2), hcat, zip(ps1, ps2))
end

# ╔═╡ 787242fc-f4c4-4b1b-bd31-6a2a68944a5b
begin 
	xlb, xub = data_lims(xps, x_sd..., x_prp..., x_fr...)
	xwb = xub .- xlb
	xlb .-= 0.05 .* xwb
	xub .+= 0.05 .* xwb
	xlb, xub
end

# ╔═╡ 6b28d8b0-d26d-475a-a6a6-3598ddd7d679
begin
	xres = 200
	
	X1 = range(xlb[1], xub[1], xres)
	X2 = range(xlb[2], xub[2], xres)

	C = [c(x1, x2) for x1=X1, x2=X2]

	nothing
end

# ╔═╡ 94da5708-9bb9-4e7d-bae5-cdb2d7e0b56b
function plot_runs!(
	fig, i, x_mats, ckey;
	critplot = nothing,
	ylabelvisible = false,
	yticklabelsvisible = ylabelvisible
)
	ax = Axis(fig[1, i];
		xlabel = L"x_1",
		ylabel = L"x_2",
		title = "$(ckey)",
		xlabelpadding = 0.5f0,
		ylabelpadding = 0.5f0,
		ylabelvisible, 
		yticklabelsvisible,
		#aspect = DataAspect(),
		#=palette = (;
			color = Makie.wong_colors()[2:end],
		)
		=#
	)

	critplot = heatmap!(
		ax, X1, X2, C; 
		colormap=:grays, colorscale=log10
	)

	lines!(
		ax, ps1, ps2; 
		color = Makie.wong_colors()[1],
		linewidth = 6f0
	)

	palette = Makie.wong_colors()[2:end]
	nC = length(palette)
	for (j, xs) in enumerate(x_mats)
		col = palette[ j % nC + 1 ]
		scatterlines!(
			ax, xs;
			color = (col, .6),
			linewidth = 1.25f0,
			markersize = 4f0
		)
	end
	for (j, xs) in enumerate(x_mats)
		col = palette[ j % nC + 1 ]
		scatter!(
			ax, xs[:, [1, end]]; 
			color = col,
			marker = [:circle, :diamond],
			markersize = 8f0,
			strokecolor = :white,
			strokewidth = 0.5f0
		)
	end
	return (; ax, critplot)
end

# ╔═╡ a3ec130a-751f-4a8b-b584-1c90b71060a8
fig = with_theme(Makie.theme_latexfonts()) do
	fig = Figure(; size = (460, 260), figure_padding=1f0)

	critplot = nothing

	for (i, (x_mats, ckey)) in enumerate((
		(x_sd, "Steepest Descent"),
		(x_prp, "PRPP"),
		(x_fr, "FRBO")
	))
		rp = plot_runs!(fig, i, x_mats, ckey;
			critplot,
			ylabelvisible = i == 1,
		)
		if isnothing(critplot)
			critplot = rp.critplot
		end
	end

	Colorbar(
		fig[2, 1:3], critplot;
		vertical = false,
		flipaxis = false,
		ticks = [10^i for i = 1:9],
		tickformat = vals -> [ l % 2 == 0 ? "" : L"10^{%$(@sprintf(\"%g\", log10(v)))}" for (l, v) in enumerate(vals)],
		minorticksvisible = true,
		minorticks = IntervalsBetween(9)
	)

	rowgap!(fig.layout, 1, 5f0)
	fig
end

# ╔═╡ 1a1f9684-0fde-4e12-a229-1dd51d08fad4
save("rb2d.png", fig; pt_per_unit=1, px_per_unit=5.0)

# ╔═╡ Cell order:
# ╠═a5953b72-bd63-11ef-1190-ffea7b2dca65
# ╠═f56ea3de-cae3-43a0-8c59-33d695b6864f
# ╠═52ecbc11-9345-4a66-bc3d-6374ae7ccf3c
# ╠═d80c68e3-d3c5-4b00-9f68-3e97a5a55daf
# ╠═5c8fe52a-84d6-49af-b859-0c9a25354d71
# ╠═0e732d21-b4a6-4dcd-ba46-c7e50c67f552
# ╠═4c788894-1a4b-4235-96a7-e3ba67f2cb77
# ╠═561269bf-2e05-42cd-a0bf-613260da63cb
# ╠═243c9ca3-85d5-41ce-9388-03c376a74fb1
# ╠═f24ad06d-2498-4ab2-830d-465571f9d18a
# ╠═a1772153-f309-4feb-81af-565b9f4a7ad5
# ╠═2749cf2a-6ebc-41d9-873b-95c7bd7c412a
# ╠═e3e5fe5e-ff63-4afc-838e-450a093895e9
# ╠═f917aeaf-e024-47ba-b405-5cdf64084c03
# ╠═5d88a5cc-0f41-4783-8173-3f3b47017eba
# ╠═d9b49cbf-2595-4b98-b82b-c1047608f768
# ╠═cd5d5aa4-4757-4afc-8b45-8c6acbccb031
# ╠═ab4e7edc-7933-499e-a80d-80e102e514c4
# ╠═cbe727f9-f211-4996-99b0-f20099ba1261
# ╠═57076e21-9ddf-43b0-9479-273e7f24ce50
# ╠═b59c9787-e1c2-4510-b069-ed8b411d3766
# ╠═6968952b-4807-4fdb-b46b-1c222081b499
# ╠═787242fc-f4c4-4b1b-bd31-6a2a68944a5b
# ╠═6b28d8b0-d26d-475a-a6a6-3598ddd7d679
# ╠═1f0e1271-4535-40e2-972d-096c472d2b5c
# ╠═a4626e6d-8692-4695-ba1b-080bf4739fa3
# ╠═94da5708-9bb9-4e7d-bae5-cdb2d7e0b56b
# ╠═a3ec130a-751f-4a8b-b584-1c90b71060a8
# ╠═1a1f9684-0fde-4e12-a229-1dd51d08fad4
