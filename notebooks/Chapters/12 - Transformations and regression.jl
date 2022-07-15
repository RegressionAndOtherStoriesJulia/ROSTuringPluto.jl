### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ d2c908b8-36a2-4add-a342-e2f0de3ba45a
using Pkg, DrWatson

# ╔═╡ d350c714-d133-46aa-a19f-129a13016fba
begin
    # Specific to this notebook
    using GLM

    # Specific to ROPTuringluto
    using Optim
    using Logging
    using Turing
    
    # Graphics related
    using GLMakie
    using Makie
    using AlgebraOfGraphics

    # Common data files and functions
	using RegressionAndOtherStories
    import RegressionAndOtherStories: link

    set_aog_theme!()
    Logging.disable_logging(Logging.Warn)
end;

# ╔═╡ d2dbce31-513b-49e9-8f03-bc622981c09f
md"## See chapter 12 in Regression and Other Stories."

# ╔═╡ 33605d56-e027-4cf6-8c40-d229bdd511df
md" ##### Widen the notebook."

# ╔═╡ 21740de8-3937-4c5f-b208-6316f3d89fc8
html"""
<style>
    main {
        margin: 0 auto;
        max-width: 2000px;
        padding-left: max(160px, 10%);
        padding-right: max(160px, 10%);
    }
</style>
"""

# ╔═╡ 00b143ae-5d99-4163-aaf3-90fae1f51d14
md"##### A typical set of Julia packages to include in notebooks."

# ╔═╡ 7f9103ac-4ee0-4836-b8d8-5a4f4773f369
md"### 12.1 "

# ╔═╡ c1e2708d-cbf1-4e7c-b276-311c3f967ea1
hdi = CSV.read(ros_datadir("HDI", "hdi.csv"), DataFrame)

# ╔═╡ Cell order:
# ╟─d2dbce31-513b-49e9-8f03-bc622981c09f
# ╟─33605d56-e027-4cf6-8c40-d229bdd511df
# ╠═21740de8-3937-4c5f-b208-6316f3d89fc8
# ╠═d2c908b8-36a2-4add-a342-e2f0de3ba45a
# ╟─00b143ae-5d99-4163-aaf3-90fae1f51d14
# ╠═d350c714-d133-46aa-a19f-129a13016fba
# ╟─7f9103ac-4ee0-4836-b8d8-5a4f4773f369
# ╠═c1e2708d-cbf1-4e7c-b276-311c3f967ea1
