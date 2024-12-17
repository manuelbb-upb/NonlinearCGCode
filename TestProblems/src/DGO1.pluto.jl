### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 09f2c3af-1343-4704-9e21-ffe7959ae40b
using PlutoDevMacros

# ╔═╡ 15ca5860-094c-44bf-b391-fb2015920ef2
@fromparent import *

# ╔═╡ a8b03a74-b867-11ef-064c-ad2fc737eb46
md"# DGO1"

# ╔═╡ e48a0e10-54a2-4232-926c-9d2967d7ed31
md"References:"

# ╔═╡ fe514db9-330b-40e6-8258-f4e908bece52
struct DGO1 <: AbstractTestProblem end

# ╔═╡ c710d4c7-aca5-46f3-aad8-eff6184b66d7
begin
	@addmethod function reference_keys(::Type{DGO1})
		return (
			"dumitrescuNewEvolutionaryApproach2000",
			"hubandReviewMultiobjectiveTest2006",
			"chenConjugateGradientMethods2024",		
		)
	end
	_dgo1_has_refs = true
end

# ╔═╡ 5555212f-712f-462f-8e27-395bccb65227
# ╠═╡ skip_as_script = true
#=╠═╡
_dgo1_has_refs && Markdown.parse(printrefs(DGO1))
  ╠═╡ =#

# ╔═╡ f35fe19c-ba76-446d-839a-665ca5dbad5f
@addmethod num_variables(::DGO1) = 1

# ╔═╡ 06fcc024-3a5c-4387-bfb2-f0db270d74c5
@addmethod num_objectives(::DGO1) = 2

# ╔═╡ 6baaa190-521f-4e7a-a889-637c56ffedf7
@addmethod function lower_variable_bounds(tp::DGO1)
	return [-10.0, ]
end

# ╔═╡ 26701ed0-1ceb-4db5-a5e2-98449edc112b
@addmethod function upper_variable_bounds(tp::DGO1)
	return [13.0, ]
end

# ╔═╡ bae8297d-a089-4777-a67c-5cc15d5c8972
@addmethod function in_place_objectives!(y, tp::DGO1, x)
	_x = x[1]
	y[1] = sin(_x)
	y[2] = sin(_x + 0.7)
    return nothing
end

# ╔═╡ ddebd250-cbe7-4b74-a72d-221f5382a357
@addmethod function in_place_jac!(Dy, tp::DGO1, x)
	_x = x[1]
	Dy[1, 1] = cos(_x)
	Dy[2, 1] = cos(_x + 0.7)
	return nothing
end

# ╔═╡ 9e9838f8-b8a4-40d5-af21-bb7c2e0b9d43
# ╠═╡ skip_as_script = true
#=╠═╡
let
	tp = DGO1()
	num_vars = num_variables(tp)
	num_objs = num_objectives(tp)
	x = ones(num_vars)
	y = zeros(num_objs)
	Dy = zeros(num_objs, num_vars)
	objectives! = IPObjectivesClosure(tp)
	jac! = IPJacClosure(tp)
	objectives!(y, x)
	jac!(Dy, x)
	y, Dy
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
# ╟─e48a0e10-54a2-4232-926c-9d2967d7ed31
# ╠═5555212f-712f-462f-8e27-395bccb65227
# ╠═fe514db9-330b-40e6-8258-f4e908bece52
# ╠═c710d4c7-aca5-46f3-aad8-eff6184b66d7
# ╠═f35fe19c-ba76-446d-839a-665ca5dbad5f
# ╠═06fcc024-3a5c-4387-bfb2-f0db270d74c5
# ╠═6baaa190-521f-4e7a-a889-637c56ffedf7
# ╠═26701ed0-1ceb-4db5-a5e2-98449edc112b
# ╠═bae8297d-a089-4777-a67c-5cc15d5c8972
# ╠═ddebd250-cbe7-4b74-a72d-221f5382a357
# ╠═9e9838f8-b8a4-40d5-af21-bb7c2e0b9d43
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002