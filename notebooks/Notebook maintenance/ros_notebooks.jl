### A Pluto.jl notebook ###
# v0.19.10

using Markdown
using InteractiveUtils

# ╔═╡ 9b0bd12f-7f0b-4386-89ff-dd39b01689b3
using Pkg

# ╔═╡ bcee8b60-3ece-4343-a88b-5bd1e00a32e2
using RegressionAndOtherStories

# ╔═╡ 67ca5484-540a-46be-8320-4bd4cf017438
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


# ╔═╡ 8c7f80e6-dc4e-11ec-3976-435b141eaf54
md" ### Using the maintenance functions."

# ╔═╡ 978ec8d1-e0c2-48b9-8a9e-feb9c9d2a465
md"

!!! note

This script is intended to be run just once."

# ╔═╡ 1bb6461a-d309-4750-82b6-f9246cd21c2e
md" ##### If a project has many Pluto notebooks, upgrading the Project and Manifest sections in each notebook currently requires you to open/run the notebook and subsequently request upgrades (e.g. click on tick mark below behind `using RegressionAndOtherStories` and click on the up-arrow) if available to be installed."

# ╔═╡ 1f6af7fd-66a6-4aff-a430-60bbcf2d127a
md" ##### For 20 substantial notebooks (as used in RegressionAndOtherStories and StatisticalRethinking derived projects) this can take 15-30 minutes!"

# ╔═╡ 8aa98ee4-c2fa-45d4-b78e-1905dc65a1c1
md" ##### Using the `update_ros_notebooks()` functions allows you to select which notebooks you would like to be updated the next time when it is opened."

# ╔═╡ 41868a6e-d959-4695-a533-9247168a83a0
md" ##### Example of the use of the `update_ros_notebooks()` function:"

# ╔═╡ 80249915-cc1c-4b6a-8740-5806cb527f01
md" ##### Change to the correct directory."

# ╔═╡ 3c22db9e-3cb2-4e53-a50d-4de35c6874f4
cd(joinpath(expanduser("~/.julia/dev/ROSTuringPluto/")))

# ╔═╡ a75b71ae-8766-4354-b71b-a6f084513645
pwd()

# ╔═╡ 6cc2b195-efc5-41d7-8f86-9926623bc64c
md" ##### Read the content of the notebooks subdirectory."

# ╔═╡ c03d53aa-a38f-4344-b476-dd0ba76bcfe2
md" ##### All notebooks in below directories can be reset as follows:"

# ╔═╡ 8baeb9bc-416c-49bb-8649-b1df05d2933c
ros_notebooks = create_ros_notebooks()

# ╔═╡ 013740dd-2310-4cd1-b762-e4db2998d05d
md" ##### Set some ros_df.reset entries to true:"

# ╔═╡ f28c46d7-a458-46f3-910a-44a09d923f4b
ros_notebooks[[1, 4, 5], :reset] .= true;

# ╔═╡ bb2c63f3-e94b-4053-a057-2f1a0f403158
ros_notebooks

# ╔═╡ 1bb688de-5c59-43c3-a2c9-e9440497c7fc
md" #### Or set all `reset` values to true:"

# ╔═╡ 7986493f-4071-4482-86ad-364ce065851d
ros_notebooks.reset .= true;

# ╔═╡ e669984e-14f0-4a76-8208-1783664c58e9
ros_notebooks

# ╔═╡ 762a1d96-021c-480d-aa0c-8969f2b60efb
update_ros_notebooks(ros_notebooks);

# ╔═╡ c3a97659-089f-40cb-853b-e2e19668d1b7
md" ##### All of these notebooks are now without Project and Manifest sections."

# ╔═╡ Cell order:
# ╠═67ca5484-540a-46be-8320-4bd4cf017438
# ╟─8c7f80e6-dc4e-11ec-3976-435b141eaf54
# ╟─978ec8d1-e0c2-48b9-8a9e-feb9c9d2a465
# ╟─1bb6461a-d309-4750-82b6-f9246cd21c2e
# ╟─1f6af7fd-66a6-4aff-a430-60bbcf2d127a
# ╟─8aa98ee4-c2fa-45d4-b78e-1905dc65a1c1
# ╟─41868a6e-d959-4695-a533-9247168a83a0
# ╠═9b0bd12f-7f0b-4386-89ff-dd39b01689b3
# ╠═bcee8b60-3ece-4343-a88b-5bd1e00a32e2
# ╟─80249915-cc1c-4b6a-8740-5806cb527f01
# ╠═3c22db9e-3cb2-4e53-a50d-4de35c6874f4
# ╠═a75b71ae-8766-4354-b71b-a6f084513645
# ╟─6cc2b195-efc5-41d7-8f86-9926623bc64c
# ╟─c03d53aa-a38f-4344-b476-dd0ba76bcfe2
# ╠═8baeb9bc-416c-49bb-8649-b1df05d2933c
# ╠═013740dd-2310-4cd1-b762-e4db2998d05d
# ╠═f28c46d7-a458-46f3-910a-44a09d923f4b
# ╠═bb2c63f3-e94b-4053-a057-2f1a0f403158
# ╟─1bb688de-5c59-43c3-a2c9-e9440497c7fc
# ╠═7986493f-4071-4482-86ad-364ce065851d
# ╠═e669984e-14f0-4a76-8208-1783664c58e9
# ╠═762a1d96-021c-480d-aa0c-8969f2b60efb
# ╟─c3a97659-089f-40cb-853b-e2e19668d1b7
