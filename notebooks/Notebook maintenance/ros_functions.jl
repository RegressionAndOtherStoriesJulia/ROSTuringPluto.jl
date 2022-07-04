### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 9b0bd12f-7f0b-4386-89ff-dd39b01689b3
using Pkg

# ╔═╡ bcee8b60-3ece-4343-a88b-5bd1e00a32e2
begin
	# Specific to this notebook
    using GLM

	# Specific to ROPTuringPluto
	using Optim
	using Logging
    using Turing
	
	# Specific to ROPStanPluto
	using StanSample
	
	# Graphics related
    using GLMakie
    using Makie
    using AlgebraOfGraphics

	# Include basic packages
	using RegressionAndOtherStories
	import RegressionAndOtherStories: link
	
	# Common data files and functions
	using RegressionAndOtherStories

	set_aog_theme!()
	Logging.disable_logging(Logging.Warn)
end;

# ╔═╡ 8c7f80e6-dc4e-11ec-3976-435b141eaf54
md" ### Function summary DataFrame."

# ╔═╡ c324d45a-1054-4bbb-a735-d4e7c2a5e0f1
md" #### Widen notebook."

# ╔═╡ d344aec5-dff4-4960-ac33-644eae5074f6
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

# ╔═╡ 41868a6e-d959-4695-a533-9247168a83a0
md" ##### Example of the use of the `reset_notebooks!()` function:"

# ╔═╡ 80249915-cc1c-4b6a-8740-5806cb527f01
md" ##### Show the contents of DataFrame `ros_functions` in this notebook."

# ╔═╡ 4dd69336-c10a-435b-aca8-cb2b40a1b135
ros_functions = update_ros_functions(create_ros_functions())

# ╔═╡ Cell order:
# ╟─8c7f80e6-dc4e-11ec-3976-435b141eaf54
# ╟─c324d45a-1054-4bbb-a735-d4e7c2a5e0f1
# ╠═d344aec5-dff4-4960-ac33-644eae5074f6
# ╟─41868a6e-d959-4695-a533-9247168a83a0
# ╠═9b0bd12f-7f0b-4386-89ff-dd39b01689b3
# ╠═bcee8b60-3ece-4343-a88b-5bd1e00a32e2
# ╟─80249915-cc1c-4b6a-8740-5806cb527f01
# ╠═4dd69336-c10a-435b-aca8-cb2b40a1b135
