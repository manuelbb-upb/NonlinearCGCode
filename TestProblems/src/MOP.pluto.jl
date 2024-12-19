### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 09f2c3af-1343-4704-9e21-ffe7959ae40b
using PlutoDevMacros

# ╔═╡ 15ca5860-094c-44bf-b391-fb2015920ef2
@fromparent import *

# ╔═╡ a8b03a74-b867-11ef-064c-ad2fc737eb46
md"# Van Veldhuizen MOP Problems"

# ╔═╡ fbd72156-f123-46f3-85f7-e9094d122dbb
abstract type AbstractMOPTestProblem <: AbstractTestProblem end

# ╔═╡ 84ad4d4d-45c8-4ea5-b880-03ac27267a09
md"References:"

# ╔═╡ b01b6509-70ad-4215-9aeb-c9a13578743c
const _common_mop_refs = (
	"hubandReviewMultiobjectiveTest2006",
	"vanVeldhuizenMOP1999",
	"chenConjugateGradientMethods2024",
)

# ╔═╡ 723b9650-eab5-454b-85cf-cf7c7cbc27c4
begin
	@addmethod function reference_keys(::Type{<:AbstractMOPTestProblem})
		return _common_mop_refs
	end
	_mop_has_refs = true
end

# ╔═╡ 9defcaaf-1b35-4e5f-b775-2ce944756fba
# ╠═╡ skip_as_script = true
#=╠═╡
_mop_has_refs && Markdown.parse(printrefs(AbstractMOPTestProblem))
  ╠═╡ =#

# ╔═╡ 737e1ee5-6657-47bf-99c8-eb52d413e974
md"## MOP 2"

# ╔═╡ fe514db9-330b-40e6-8258-f4e908bece52
Base.@kwdef struct MOP2 <: AbstractMOPTestProblem 
	L :: Float64 = -4.0
	U :: Float64 = 4.0
end

# ╔═╡ abca6c2f-30ff-4bdf-aeba-8a928726359d
begin
	@addmethod num_variables(::MOP2) = 2
	@addmethod num_objectives(::MOP2) = 2
	@addmethod function lower_variable_bounds(tp::MOP2)
		return fill(tp.L, 2)
	end
	@addmethod function upper_variable_bounds(tp::MOP2)
		return fill(tp.U, 2)
	end
	@addmethod function reference_keys(::Type{MOP2})
		return (
			"fonsecaMultiobjectiveGeneticAlgorithms1995",
			_common_mop_refs...
		)
	end
end

# ╔═╡ bae8297d-a089-4777-a67c-5cc15d5c8972
@addmethod function in_place_objectives!(y, tp::MOP2, x)
	n = num_variables(tp)
	sn = 1/sqrt(n)
	y[1] = 1 - exp( - sum( (x .- sn).^2 ) )
	y[2] = 1 - exp( - sum( (x .+ sn).^2 ) )
    return nothing
end

# ╔═╡ 83e40115-b9e5-4569-9d15-07ed49088f4a
@addmethod function in_place_jac!(Dy, tp::MOP2, x)
	Dy .= 0
	n = num_variables(tp)
	sn = 1/sqrt(n)

	Dy[1, :] = (2 * exp( - sum( (x .- sn).^2 ) )) .* (x .- sn)
	Dy[2, :] = (2 * exp( - sum( (x .+ sn).^2 ) )) .* (x .+ sn)
	
    return nothing
end

# ╔═╡ 9e9838f8-b8a4-40d5-af21-bb7c2e0b9d43
# ╠═╡ skip_as_script = true
#=╠═╡
let	tp = MOP2();
	
	num_vars = num_variables(tp)
	num_objs = num_objectives(tp)

	lb = lower_variable_bounds(tp)
	ub = upper_variable_bounds(tp)
	
	x = lb .+ (ub .- lb) .* rand(num_vars)
	
	y = zeros(num_objs)
	Dy = zeros(num_objs, num_vars)
	objectives! = IPObjectivesClosure(tp)
	jac! = IPJacClosure(tp)
	objectives!(y, x)
	jac!(Dy, x)
	
	Dy .- fd_jac(tp, x)
end
  ╠═╡ =#

# ╔═╡ 41a8aa2f-ae92-4ca7-ad6c-f7449d7a1d21
md"## MOP3"

# ╔═╡ 6f8fb268-bf8f-46cd-ba83-14d65a12e8d5
struct MOP3 <: AbstractMOPTestProblem end

# ╔═╡ 1f164705-2181-4369-ba70-ae488fc8b7cb
begin
	@addmethod num_variables(::MOP3) = 2
	@addmethod num_objectives(::MOP3) = 2
	@addmethod function lower_variable_bounds(tp::MOP3)
		return [-π, -π]
	end
	@addmethod function upper_variable_bounds(tp::MOP3)
		return -lower_variable_bounds(tp)
	end
	@addmethod function reference_keys(::Type{MOP3})
		return (
			"poloni1996multi",
			_common_mop_refs...
		)
	end
end

# ╔═╡ 1039c66f-7fb1-451b-bf9e-b8b512370af2
@addmethod function in_place_objectives!(y, tp::MOP3, x)
	n = num_variables(tp)
	sin1 = sin(1)
	cos1 = cos(1)
	sin2 = sin(2)
	cos2 = cos(2)
	A1 = 0.5 * sin1 - 2 * cos1 + sin2 - 1.5 * cos2
	A2 = 0.5 * sin1 - cos1 + 2 * sin2 - 0.5 * cos2
	B1 = 0.5 * sin(x[1]) - 2 * cos(x[1]) + sin(x[2]) - 1.5 * cos(x[2])
	B2 = 1.5 * sin(x[1]) - 1 * cos(x[1]) + 2 * sin(x[2]) - 0.5 * cos(x[2])

	y[1] = -1 - (A1 - B1)^2 - (A2 - B2)^2 
	y[2] = -(x[1] + 3)^2 - (x[2] + 1)^2 

	y .*= -1
    return nothing
end

# ╔═╡ 7ce76189-4024-4857-b7e7-ac6cdab3a27d
@addmethod function in_place_jac!(Dy, tp::MOP3, x)
	Dy .= 0
	sin1 = sin(1)
	cos1 = cos(1)
	sin2 = sin(2)
	cos2 = cos(2)
	A1 = 0.5 * sin1 - 2 * cos1 + sin2 - 1.5 * cos2
	A2 = 0.5 * sin1 - cos1 + 2 * sin2 - 0.5 * cos2
	B1 = 0.5 * sin(x[1]) - 2 * cos(x[1]) + 1 * sin(x[2]) - 1.5 * cos(x[2])
	B2 = 1.5 * sin(x[1]) - 1 * cos(x[1]) + 2 * sin(x[2]) - 0.5 * cos(x[2])

	B1_dx1 = 0.5 * cos(x[1]) + 2.0 * sin(x[1])
	B1_dx2 = 1.0 * cos(x[2]) + 1.5 * sin(x[2])
	B2_dx1 = 1.5 * cos(x[1]) + 1.0 * sin(x[1])
	B2_dx2 = 2.0 * cos(x[2]) + 0.5 * sin(x[2])

	Dy[1, 1] = 2 * (A1 - B1) * B1_dx1
	Dy[1, 2] = 2 * (A1 - B1) * B1_dx2
	
	Dy[1, 1] += 2 * (A2 - B2) * B2_dx1
	Dy[1, 2] += 2 * (A2 - B2) * B2_dx2

	Dy[2, 1] = -2 * (x[1] + 3)
	Dy[2, 2] = -2 * (x[2] + 1)

	Dy .*= -1
	
    return nothing
end

# ╔═╡ 4f72a687-c79c-40fc-a398-68193314d442
# ╠═╡ skip_as_script = true
#=╠═╡
let	tp = MOP3();
	
	num_vars = num_variables(tp)
	num_objs = num_objectives(tp)

	lb = lower_variable_bounds(tp)
	ub = upper_variable_bounds(tp)
	
	x = lb .+ (ub .- lb) .* rand(num_vars)
	
	y = zeros(num_objs)
	Dy = zeros(num_objs, num_vars)
	objectives! = IPObjectivesClosure(tp)
	jac! = IPJacClosure(tp)
	objectives!(y, x)
	jac!(Dy, x)
	
	Dy .- fd_jac(tp, x)
end
  ╠═╡ =#

# ╔═╡ decbc69b-ba0a-4673-b322-bb5412076b9e
md"## MOP5"

# ╔═╡ b320506f-ce7d-4c77-a03c-db04e93f8342
struct MOP5 <: AbstractMOPTestProblem end

# ╔═╡ 3c3c3a22-3e5a-4bc1-9864-84a585580e4e
begin
	@addmethod num_variables(::MOP5) = 2
	@addmethod num_objectives(::MOP5) = 3
	@addmethod function lower_variable_bounds(tp::MOP5)
		return fill(-30.0, num_variables(tp))
	end
	@addmethod function upper_variable_bounds(tp::MOP5)
		return -lower_variable_bounds(tp)
	end
	@addmethod function reference_keys(::Type{MOP5})
		return (
			"poloni1996multi",
			_common_mop_refs...
		)
	end
end

# ╔═╡ d1f17fea-6dbe-4d72-8622-913990cab453
@addmethod function in_place_objectives!(y, tp::MOP5, x)
	xsq = sum( x.^2 )
	y[1] = 0.5 * xsq + sin( xsq )
	y[2] = (3 * x[1] - 2 * x[2] + 4)^2 / 8 + (x[1] - x[2] + 1)^2 / 27 + 15
	y[3] = 1 / (xsq + 1) - 1.1 * exp( - xsq )
	return nothing
end

# ╔═╡ 90668f19-a8ab-4941-b790-71bf1fab2e67
@addmethod function in_place_jac!(Dy, tp::MOP5, x)
	Dy .= 0

	xsq = sum( x.^2 )
	#  0.5 * xsq + sin( xsq )
	Dy[1, :] .= x
	Dy[1, :] .+= cos(xsq) * 2 .* x
	
	# (3 * x[1] - 2 * x[2] + 4)^2 / 8 + (x[1] - x[2] + 1)^2 / 27 + 15
	a = 2 * (3 * x[1] - 2 * x[2] + 4) / 8
	b = 2 * (x[1] - x[2] + 1) / 27
	Dy[2, 1] = 3 * a + b
	Dy[2, 2] = -2 * a - b

	# 1 / (xsq + 1) - 1.1 * exp( - xsq )
	Dy[3, :] = ( -2/(xsq + 1)^2 - 2 * exp( -xsq ) ) .* x
	return nothing
end

# ╔═╡ 3782a63d-5a50-456f-a807-730b9f904c70
# ╠═╡ skip_as_script = true
#=╠═╡
let	tp = MOP5();
	
	num_vars = num_variables(tp)
	num_objs = num_objectives(tp)

	lb = lower_variable_bounds(tp)
	ub = upper_variable_bounds(tp)
	
	x = lb .+ (ub .- lb) .* rand(num_vars)
	
	y = zeros(num_objs)
	Dy = zeros(num_objs, num_vars)
	objectives! = IPObjectivesClosure(tp)
	jac! = IPJacClosure(tp)
	objectives!(y, x)
	jac!(Dy, x)
	
	Dy .- fd_jac(tp, x)
end
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"

[compat]
PlutoDevMacros = "~0.9.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "9c51cf46a1e177639a0c248b9cdccbfd7b4d05d7"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "10da5154188682e5c0726823c2b5125957ec3778"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.38"

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

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoDevMacros]]
deps = ["JuliaInterpreter", "Logging", "MacroTools", "Pkg", "TOML"]
git-tree-sha1 = "72f65885168722413c7b9a9debc504c7e7df7709"
uuid = "a0499f29-c39b-4c5c-807c-88074221b949"
version = "0.9.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═09f2c3af-1343-4704-9e21-ffe7959ae40b
# ╠═15ca5860-094c-44bf-b391-fb2015920ef2
# ╟─a8b03a74-b867-11ef-064c-ad2fc737eb46
# ╠═fbd72156-f123-46f3-85f7-e9094d122dbb
# ╟─84ad4d4d-45c8-4ea5-b880-03ac27267a09
# ╠═9defcaaf-1b35-4e5f-b775-2ce944756fba
# ╠═b01b6509-70ad-4215-9aeb-c9a13578743c
# ╠═723b9650-eab5-454b-85cf-cf7c7cbc27c4
# ╟─737e1ee5-6657-47bf-99c8-eb52d413e974
# ╠═fe514db9-330b-40e6-8258-f4e908bece52
# ╠═abca6c2f-30ff-4bdf-aeba-8a928726359d
# ╠═bae8297d-a089-4777-a67c-5cc15d5c8972
# ╠═83e40115-b9e5-4569-9d15-07ed49088f4a
# ╠═9e9838f8-b8a4-40d5-af21-bb7c2e0b9d43
# ╟─41a8aa2f-ae92-4ca7-ad6c-f7449d7a1d21
# ╠═6f8fb268-bf8f-46cd-ba83-14d65a12e8d5
# ╠═1f164705-2181-4369-ba70-ae488fc8b7cb
# ╠═1039c66f-7fb1-451b-bf9e-b8b512370af2
# ╠═7ce76189-4024-4857-b7e7-ac6cdab3a27d
# ╠═4f72a687-c79c-40fc-a398-68193314d442
# ╟─decbc69b-ba0a-4673-b322-bb5412076b9e
# ╠═b320506f-ce7d-4c77-a03c-db04e93f8342
# ╠═3c3c3a22-3e5a-4bc1-9864-84a585580e4e
# ╠═d1f17fea-6dbe-4d72-8622-913990cab453
# ╠═90668f19-a8ab-4941-b790-71bf1fab2e67
# ╠═3782a63d-5a50-456f-a807-730b9f904c70
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
