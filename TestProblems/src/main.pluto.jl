### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 1550f0f2-69de-4e56-b9ed-5fe5294300cb
import LinearAlgebra as LA

# ╔═╡ db4323b2-285e-4ecc-a7e4-24a053554270
md"# Interface"

# ╔═╡ 4c1d91ed-5c01-431e-ab4f-bf7ba5862b8a
abstract type AbstractTestProblem end

# ╔═╡ a08ddd38-122e-4054-871d-6c0aaa173dae
num_variables(::AbstractTestProblem)::Integer=0

# ╔═╡ bae1f330-4c9f-4779-bd2f-c4534225e09d
num_objectives(::AbstractTestProblem)::Integer=0

# ╔═╡ 32879755-af83-4718-9f10-7297f0ac72d4
function lower_variable_bounds(tp::AbstractTestProblem)
    return error("`method not implemented for $(typeof(tp)).")
end

# ╔═╡ 519c0921-016d-4e06-89ec-24dbd6ab52e7
function upper_variable_bounds(tp::AbstractTestProblem)
    return error("`method not implemented for $(typeof(tp)).")
end

# ╔═╡ 67dcab82-8209-4f7b-bf63-cacebeaaedb2
abstract type AbstractTestProblemError end

# ╔═╡ 5dd6b5b2-d19c-46be-921c-cb8c1b2e4bcd
function in_place_objectives!(y, tp::AbstractTestProblem, x)
    return error("`method not implemented for $(typeof(tp)).")
end

# ╔═╡ 02c4f00f-712d-497a-8a4a-466a415519e8
function in_place_jac!(Dy, tp::AbstractTestProblem, x)
    return error("`method not implemented for $(typeof(tp)).")
end

# ╔═╡ fc1fabd6-2cf1-47e7-8c7a-42d3ea04f4e2
function reference_keys(::Type{<:AbstractTestProblem})
	nothing
end

# ╔═╡ 7417770a-0238-4835-af0f-f0d38e832aff
function reference_keys(tp::AbstractTestProblem)
	return reference_keys(typeof(tp))
end

# ╔═╡ c48a893a-d2a3-4ce7-acd0-5b7c4cf5cebe
md"## Scaling"

# ╔═╡ 1c089573-f156-4964-8f50-a256f381cf2b
struct RScaledTestProblem{
    tp_Type
} <: AbstractTestProblem
    tp :: tp_Type
    r :: Vector{Float64}
end

# ╔═╡ daa89ac9-3970-438a-be25-45f4976fee32
num_variables(rtp::RScaledTestProblem)=num_variables(rtp.tp)

# ╔═╡ 523afe23-6ac0-4d9d-9fce-28fc510d6dfb
num_objectives(rtp::RScaledTestProblem)=num_objectives(rtp.tp)

# ╔═╡ 84c1ef9c-016a-44f1-bf84-895f1fc7fb34
function in_place_objectives!(y, rtp::RScaledTestProblem, x)
    tp = rtp.tp
    r = rtp.r
    in_place_objectives!(y, tp, x)
    y .*= r
    return nothing
end

# ╔═╡ 99e6a169-cda6-4cdf-9dd2-604602a3e4fd
begin
struct IPObjectivesClosure{
    tp_Type
}
    tp :: tp_Type
end
function (oc::IPObjectivesClosure)(y, x)
    tp = oc.tp
    return in_place_objectives!(y, tp, x)
end
end

# ╔═╡ 6db0e549-182e-499a-a3ea-aa0f3facb554
begin
struct OOPObjectivesClosure{
    tp_Type
}
    tp :: tp_Type
end
function (oc::OOPObjectivesClosure)(x)
    tp = oc.tp
	no = num_objectives(tp)
	y = zeros(no)
    code = in_place_objectives!(y, tp, x)
	if isa(code, AbstractTestProblemError)
		return code
	end
	return y
end
end

# ╔═╡ 0daa6bb3-55e0-4863-8962-c1953fc3822d
function in_place_jac!(Dy, rtp::RScaledTestProblem, x)
    tp = rtp.tp
    r = rtp.r
    in_place_jac!(Dy, tp, x)
    Dy .*= r
    return nothing
end

# Finite Difference Jacobian for testing

# ╔═╡ 6189f4ac-5fdb-4f2e-bd54-97cf7ac120e0
begin
struct IPJacClosure{
    tp_Type
}
    tp :: tp_Type
end
function (oc::IPJacClosure)(Dy, x)
    tp = oc.tp
    return in_place_jac!(Dy, tp, x)
end
end

# ╔═╡ d768b7ca-85d5-4687-a558-9678f08bef70
begin
struct OOPJacClosure{
    tp_Type
}
    tp :: tp_Type
end
function (oc::OOPJacClosure)(x)
    tp = oc.tp
	nv = num_variables(tp)
	no = num_objectives(tp)
	Dy = zeros(no, nv)
    code = in_place_jac!(Dy, tp, x)
	if code isa AbstractTestProblemError
		return code
	end
	return Dy
end
end

# ╔═╡ a5c9bd8b-2a0b-4711-8317-e96ab67198e2
function make_scaled(tp::AbstractTestProblem, x0)
    Dy = zeros(num_objectives(tp), num_variables(tp))
    in_place_jac!(Dy, tp, x0)
    r = ones(num_objectives(tp))
    for (i, g) = enumerate(eachrow(Dy))
        r[i] /= max(1, LA.norm(g, Inf))
    end
    return RScaledTestProblem(tp, r)
end

# ╔═╡ 9265efc0-1032-4d88-89ff-ede99bddaceb
md"## Utils"

# ╔═╡ 07005971-1944-4894-a485-ac24036ee6ab
function fd_jac(tp, x; h = 1e-6)
    num_vars = num_variables(tp)
    num_objs = num_objectives(tp)

    y1 = zeros(num_objs)
    y2 = similar(y1)
    ξ1 = zeros(num_vars)
    ξ2 = similar(ξ1)

    Dy = zeros(num_objs, num_vars)

    objectives! = IPObjectivesClosure(tp)

    for i = eachindex(x)
        ξ1 .= x
        ξ2 .= x
        ξ1[i] += h/2
        ξ2[i] -= h/2
        objectives!(y1, ξ1)
        objectives!(y2, ξ2)
        Dy[:, i] .= (y1 - y2) / h
        end

    return Dy
end

# ╔═╡ 28d8006c-5a26-4a0f-869f-61b73de91b0f
md"## References"

# ╔═╡ 0be34e56-eaef-4d9f-ad74-7df389fc2af3
import Bibliography:import_bibtex

# ╔═╡ 4c928811-ff94-47a8-95d8-dd38e53aa932
import DocumenterCitations: format_bibliography_reference

# ╔═╡ 415fdb63-87e4-495b-9b0e-2a084b183aca
global_bib = import_bibtex(joinpath("..", "references.bib"); check=:none);

# ╔═╡ 85edd709-ae0d-422a-82cb-6c418a64402d
function pkgref(k; verbatim=true)
	global global_bib
	entry = get(global_bib, k, nothing)
	refstr = if isnothing(entry)
		"REFERENCE `$(k)` NOT FOUND"
	else
		format_bibliography_reference(:authoryear, entry)
	end
	if verbatim
		refstr = "`"*refstr*"`"
	end
	return refstr
end

# ╔═╡ 9a579a3e-36b6-40ac-8c23-a28a89cbb38b
function printrefs(tp; kwargs...)
	_k = reference_keys(tp)
	isnothing(_k) && return "NO REFERENCE"
	return join([pkgref(k; kwargs...) for k in _k], "\n\n")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Bibliography = "f1be7e48-bf82-45af-a471-ae754a193061"
DocumenterCitations = "daee34ce-89f3-4625-b898-19384cb65244"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[compat]
Bibliography = "~0.2.20"
DocumenterCitations = "~1.3.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "3bf6ece5fc975f4ae3a5bec5e754df2cda2c63ae"

[[deps.ANSIColoredPrinters]]
git-tree-sha1 = "574baf8110975760d391c710b6341da1afa48d8c"
uuid = "a4c015fc-c6ff-483c-b24f-f7ea428134e9"
version = "0.0.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BibInternal]]
deps = ["TestItems"]
git-tree-sha1 = "b3107800faf461eca3281f89f8d768f4b3e99969"
uuid = "2027ae74-3657-4b95-ae00-e2f7d55c3e64"
version = "0.3.7"

[[deps.BibParser]]
deps = ["BibInternal", "DataStructures", "Dates", "JSONSchema", "TestItems", "YAML"]
git-tree-sha1 = "33478bed83bd124ea8ecd9161b3918fb4c70e529"
uuid = "13533e5b-e1c2-4e57-8cef-cac5e52f6474"
version = "0.2.2"

[[deps.Bibliography]]
deps = ["BibInternal", "BibParser", "DataStructures", "Dates", "FileIO", "YAML"]
git-tree-sha1 = "520c679daed011ce835d9efa7778863aad6687ed"
uuid = "f1be7e48-bf82-45af-a471-ae754a193061"
version = "0.2.20"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

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

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Documenter]]
deps = ["ANSIColoredPrinters", "AbstractTrees", "Base64", "CodecZlib", "Dates", "DocStringExtensions", "Downloads", "Git", "IOCapture", "InteractiveUtils", "JSON", "LibGit2", "Logging", "Markdown", "MarkdownAST", "Pkg", "PrecompileTools", "REPL", "RegistryInstances", "SHA", "TOML", "Test", "Unicode"]
git-tree-sha1 = "d0ea2c044963ed6f37703cead7e29f70cba13d7e"
uuid = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
version = "1.8.0"

[[deps.DocumenterCitations]]
deps = ["AbstractTrees", "Bibliography", "Dates", "Documenter", "Logging", "Markdown", "MarkdownAST", "OrderedCollections", "Unicode"]
git-tree-sha1 = "5a72f3f804deb1431cb79f5636163e4fdf8ed8ed"
uuid = "daee34ce-89f3-4625-b898-19384cb65244"
version = "1.3.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e51db81749b0777b2147fbe7b783ee79045b8e99"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.4+1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.Git]]
deps = ["Git_jll"]
git-tree-sha1 = "04eff47b1354d702c3a85e8ab23d539bb7d5957e"
uuid = "d7ba0133-e1db-5d97-8f8c-041e4b3a1eb2"
version = "1.3.1"

[[deps.Git_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "LibCURL_jll", "Libdl", "Libiconv_jll", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "399f4a308c804b446ae4c91eeafadb2fe2c54ff9"
uuid = "f8c6e375-362e-5223-8a59-34ff63f689eb"
version = "2.47.1+0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

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

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "1d322381ef7b087548321d3f878cb4c9bd8f8f9b"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.1"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JSONSchema]]
deps = ["Downloads", "JSON", "JSON3", "URIs"]
git-tree-sha1 = "243f1cdb476835d7c249deb9f29ad6b7827da7d3"
uuid = "7d188eb4-7ad8-530c-ae41-71a32a6d4692"
version = "1.4.1"

[[deps.LazilyInitializedFields]]
git-tree-sha1 = "0f2da712350b020bc3957f269c9caad516383ee0"
uuid = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
version = "1.3.0"

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

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "61dfdba58e585066d8bce214c5a51eaa0539f269"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MarkdownAST]]
deps = ["AbstractTrees", "Markdown"]
git-tree-sha1 = "465a70f0fc7d443a00dcdc3267a497397b8a3899"
uuid = "d0879d2d-cac2-40c8-9cee-1863dc0c7391"
version = "0.1.2"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+1"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

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

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RegistryInstances]]
deps = ["LazilyInitializedFields", "Pkg", "TOML", "Tar"]
git-tree-sha1 = "ffd19052caf598b8653b99404058fce14828be51"
uuid = "2792f1a3-b283-48e8-9a74-f99dce5104f3"
version = "0.1.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "b765e46ba27ecf6b44faf70df40c57aa3a547dcb"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.7"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TestItems]]
git-tree-sha1 = "42fd9023fef18b9b78c8343a4e2f3813ffbcefcb"
uuid = "1c621080-faea-4a02-84b6-bbd5e436b8fe"
version = "1.0.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.YAML]]
deps = ["Base64", "Dates", "Printf", "StringEncodings"]
git-tree-sha1 = "dea63ff72079443240fbd013ba006bcbc8a9ac00"
uuid = "ddb6d928-2868-570f-bddf-ab3f9cf99eb6"
version = "0.4.12"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

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
# ╠═1550f0f2-69de-4e56-b9ed-5fe5294300cb
# ╟─db4323b2-285e-4ecc-a7e4-24a053554270
# ╠═4c1d91ed-5c01-431e-ab4f-bf7ba5862b8a
# ╠═a08ddd38-122e-4054-871d-6c0aaa173dae
# ╠═bae1f330-4c9f-4779-bd2f-c4534225e09d
# ╠═32879755-af83-4718-9f10-7297f0ac72d4
# ╠═519c0921-016d-4e06-89ec-24dbd6ab52e7
# ╠═67dcab82-8209-4f7b-bf63-cacebeaaedb2
# ╠═5dd6b5b2-d19c-46be-921c-cb8c1b2e4bcd
# ╠═02c4f00f-712d-497a-8a4a-466a415519e8
# ╠═99e6a169-cda6-4cdf-9dd2-604602a3e4fd
# ╠═6db0e549-182e-499a-a3ea-aa0f3facb554
# ╠═6189f4ac-5fdb-4f2e-bd54-97cf7ac120e0
# ╠═d768b7ca-85d5-4687-a558-9678f08bef70
# ╠═fc1fabd6-2cf1-47e7-8c7a-42d3ea04f4e2
# ╠═7417770a-0238-4835-af0f-f0d38e832aff
# ╟─c48a893a-d2a3-4ce7-acd0-5b7c4cf5cebe
# ╠═1c089573-f156-4964-8f50-a256f381cf2b
# ╠═a5c9bd8b-2a0b-4711-8317-e96ab67198e2
# ╠═daa89ac9-3970-438a-be25-45f4976fee32
# ╠═523afe23-6ac0-4d9d-9fce-28fc510d6dfb
# ╠═84c1ef9c-016a-44f1-bf84-895f1fc7fb34
# ╠═0daa6bb3-55e0-4863-8962-c1953fc3822d
# ╟─9265efc0-1032-4d88-89ff-ede99bddaceb
# ╠═07005971-1944-4894-a485-ac24036ee6ab
# ╟─28d8006c-5a26-4a0f-869f-61b73de91b0f
# ╠═0be34e56-eaef-4d9f-ad74-7df389fc2af3
# ╠═4c928811-ff94-47a8-95d8-dd38e53aa932
# ╠═415fdb63-87e4-495b-9b0e-2a084b183aca
# ╠═85edd709-ae0d-422a-82cb-6c418a64402d
# ╠═9a579a3e-36b6-40ac-8c23-a28a89cbb38b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
