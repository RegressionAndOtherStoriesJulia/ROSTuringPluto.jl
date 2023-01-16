### A Pluto.jl notebook ###
# v0.19.10

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ be6b0ea9-d8d2-43fe-b33b-c5d136939468
using Pkg, DrWatson

# ╔═╡ 7cb960bc-cc80-47ca-8814-926be641b74d
begin
	# Specific to this notebook
    using GLM
    using PlutoUI

	# Specific to ROSTuringPluto
	using Optim
	using Logging
    using Turing
	
	# Graphics related
    using GLMakie

	# Common data files and functions
	using RegressionAndOtherStories
	import RegressionAndOtherStories: link

	Logging.disable_logging(Logging.Warn)
end;

# ╔═╡ e1fb3340-fa2d-42a0-b45a-91d8049aa373
md"## See chapter 5 in Regression and Other Stories."

# ╔═╡ 2aa720d8-717c-4fbd-9dfc-a50872c30b0b
md" ###### Widen the notebook."

# ╔═╡ a107a87f-dc67-4aa8-8f17-4ae25bc2785b
# ed172871-fa4d-4111-ac0a-341898917948
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

# ╔═╡ d955bd3e-eeb1-4c52-b01d-2577a4af2a88
md"###### A typical set of Julia packages to include in notebooks."

# ╔═╡ 0d55b83b-cea9-432d-8750-9796c0ef7c77
md" ### 5.1 Simulations of discrete events."

# ╔═╡ 8d3093a3-ee6d-49d3-bf3f-1271d4eab4e1
@bind nsim PlutoUI.Slider(2:5, default=3)

# ╔═╡ 32a81ce8-dbd3-4fb0-84cd-a121dedeb3ce
nsim

# ╔═╡ 0d6ac50d-19d8-45b5-b3e8-f4d838eea0a9
let
	f = Figure()
	ax = Axis(f[1, 1]; xlabel="n_girls", ylabel="Frequency")
	n_girls = rand(Binomial(400, 0.488), 10^nsim)
	hist!(n_girls; strokewidth = 1, strokecolor = :black)
	f
end

# ╔═╡ 6580cb29-1bf8-4c61-8619-d2822ac71ac3
function prob_girls(bt) 
	res = if bt == :single_birth
		rand(Binomial(1, 0.488), 1)
	elseif bt == :fraternal_twin
		2rand(Binomial(1, 0.495), 1)
	else
		rand(Binomial(2, 0.495), 1)
	end
	return res[1]
end

# ╔═╡ 8ecfe5aa-e8c2-42c7-b249-2ead81db1e78
function girls(no_of_births = 400;
		birth_types = [:fraternal_twin, :identical_twin, :single_birth],
		probabilities = [1/125, 1/300, 1 - 1/125 - 1/300])
	
	return prob_girls.(sample(birth_types, Weights(probabilities), no_of_births))
end

# ╔═╡ 4f4cbe49-cde1-400a-a16b-b152f814b2f8
girls()

# ╔═╡ 8f330ec4-1e08-4137-ab9d-7840486ff708
sum(girls())

# ╔═╡ c7506bc7-f978-45ab-a5f0-41142b0a45d5
let
	#Random.seed!(1)
	f = Figure()
	ax = Axis(f[1, 1]; xlabel="n_girls", ylabel="Frequency")
	girls_sim = [sum(girls()) for i in 1:1000]
	hist!(f[1, 1], girls_sim; strokewidth = 1, strokecolor = :black, xlabel="Girls")
	f
end	

# ╔═╡ 9f3d53be-5b0c-4ae2-a05a-8be752d3a7c2
md" ### 5.2 Simulation of continuous and mixed/continuous models."

# ╔═╡ 60f4fe61-ad12-4a7e-a88e-97c461658d77
let
	n_sims = 1000
	y1 = rand(Normal(3, 0.5), n_sims)
	y2 = [Exponential(y1[i]).θ for i in 1:length(y1)]
	y3 = rand(Binomial(20, 0.5), n_sims)
	y4 = rand(Poisson(5), n_sims)

	f = Figure()
	ax = Axis(f[1, 1]; title="1000 draws from Normal(3, 0.5)", xlabel="n_girls", ylabel="Frequency")
	hist!(y1; bins=20)
	ax = Axis(f[1, 2]; title="1000 draws from Exponential(y1)", xlabel="n_girls", ylabel="Frequency")
	hist!(y2; bins=20)

	ax = Axis(f[2, 1]; title="1000 draws from Binomial(20, 0.5", xlabel="n_girls", ylabel="Frequency")
	hist!(y3; bins=15)

	ax = Axis(f[2, 2]; title="1000 draws from Poisson(5)", xlabel="n_girls", ylabel="Frequency")
	hist!(y4; bins=10)
	f
end

# ╔═╡ 5726ebb6-4f60-487b-9bea-4786856c8fba
function sim()
	N = 10
	male = rand(Binomial(1, 0.48), N)
	height = male == 1 ? rand(Normal(69.1, 2.9), N) : rand(Normal(63.7, 2.7), N)
	avg_height = mean(height)
end

# ╔═╡ 219ab907-b09c-475f-90d0-1ab243e26b77
sim()

# ╔═╡ 5cbc01bb-f9db-4004-a8cc-428b9b5b994c
let
	n_sim = 1000
	avg_height = Float64[]
	for i in 1:n_sim
		append!(avg_height, [sim()])
	end
	hist(avg_height)
end

# ╔═╡ 9240436a-eca2-495a-a074-f32fd0c14c34
md" ### 5.3 Summarizing a set of simulations using median and median absolute deviation."

# ╔═╡ 28ded2b7-0fad-47d4-a5ec-04bf98284f01
let
	N = 10000
	z = rand(Normal(5, 2), N)
	vals = round.([mean(z), median(z), std(z), mad(z), 1.483 .* median(abs.(z .- median(z))), std(z)/sqrt(N)]; digits=2)
	(mean=vals[1], median=vals[2], std=vals[3], mad=vals[4], mad_sd=vals[5], std_mean = vals[6])
end

# ╔═╡ 960dc301-0e7b-4908-bbc1-10cb0190c3f2
md" ###### Standard deviation of the mean:"

# ╔═╡ 049272e7-c8cc-4fc3-9fdc-fc992aaab814
quantile(rand(Normal(5, 2), 10000), [0.25, 0.75])

# ╔═╡ 4f1eff47-8430-462c-822a-73bce9d1a186
md" ### 5.4 Bootstrapping to simulate a sampling distribution."

# ╔═╡ 562dbff1-c0c7-4d13-92ce-8274d0da315f
earnings = CSV.read(ros_datadir("Earnings", "earnings.csv"), DataFrame)

# ╔═╡ c8073f02-3e1f-4417-b330-3c720c33e31d
ratio = median(earnings[earnings.male .== 0, :earn]) /  median(earnings[earnings.male .== 1, :earn])

# ╔═╡ cf98008f-47e9-455b-b124-f235a89912db
function take_df_sample(df, size; replace = true, ordered = true)
    df[sample(axes(df, 1), size; replace, ordered), :]
end

# ╔═╡ f828c4d4-5938-4b8b-ae0e-b7bcb3f3afe6
take_df_sample(earnings, 3)

# ╔═╡ 958d4c35-bcfb-4443-99ee-720b75399a57
function boot_ratio(df::DataFrame, sym::Symbol; draws=1000, replace=true)
	df = take_df_sample(df, draws; replace)
	ratio = median(df[df.male .== 0, sym]) /  median(df[df.male .== 1, sym])
end

# ╔═╡ 32bf9f83-a45a-49d5-9d20-5d46ec6f4288
take_df_sample(earnings, 10)

# ╔═╡ b837eb3b-5819-49aa-b503-f6c4eb6b153c
boot_ratio(earnings, :earn; draws=5)

# ╔═╡ a3bcd304-03d8-46fc-bc7a-cc8e5f5bb120
let
	n_sims = 10000
	global boot_output = [boot_ratio(earnings, :earn; draws=500) for _ in 1:n_sims]
	hist(boot_output)
end

# ╔═╡ 661785d2-9302-45b9-9b86-d74903ec895a
boot_output

# ╔═╡ 45440a46-a3ec-43e7-8033-b88dbb4776ca
std(boot_output)

# ╔═╡ 59a9e17d-e0a9-4a43-a82b-5b7884b6d2f4
md" ### 5.5 Fake-data simulations as a way of life."

# ╔═╡ e36b3ba3-e922-499c-a87d-2895ced1e6f3
md" ###### Not done yet."

# ╔═╡ c2eada85-41c4-46a1-ad52-a39230a73ca7
md"
!!! note

Quick math notation test."

# ╔═╡ e4866712-03b7-4e9f-b52b-c169def8533e
md"""
```math
    \begin{cases}
      x+y=7\\
      -x+3y=1
    \end{cases}
```
"""

# ╔═╡ Cell order:
# ╟─e1fb3340-fa2d-42a0-b45a-91d8049aa373
# ╟─2aa720d8-717c-4fbd-9dfc-a50872c30b0b
# ╠═a107a87f-dc67-4aa8-8f17-4ae25bc2785b
# ╠═be6b0ea9-d8d2-43fe-b33b-c5d136939468
# ╟─d955bd3e-eeb1-4c52-b01d-2577a4af2a88
# ╠═7cb960bc-cc80-47ca-8814-926be641b74d
# ╟─0d55b83b-cea9-432d-8750-9796c0ef7c77
# ╠═8d3093a3-ee6d-49d3-bf3f-1271d4eab4e1
# ╠═32a81ce8-dbd3-4fb0-84cd-a121dedeb3ce
# ╠═0d6ac50d-19d8-45b5-b3e8-f4d838eea0a9
# ╠═6580cb29-1bf8-4c61-8619-d2822ac71ac3
# ╠═8ecfe5aa-e8c2-42c7-b249-2ead81db1e78
# ╠═4f4cbe49-cde1-400a-a16b-b152f814b2f8
# ╠═8f330ec4-1e08-4137-ab9d-7840486ff708
# ╠═c7506bc7-f978-45ab-a5f0-41142b0a45d5
# ╟─9f3d53be-5b0c-4ae2-a05a-8be752d3a7c2
# ╠═60f4fe61-ad12-4a7e-a88e-97c461658d77
# ╠═5726ebb6-4f60-487b-9bea-4786856c8fba
# ╠═219ab907-b09c-475f-90d0-1ab243e26b77
# ╠═5cbc01bb-f9db-4004-a8cc-428b9b5b994c
# ╟─9240436a-eca2-495a-a074-f32fd0c14c34
# ╠═28ded2b7-0fad-47d4-a5ec-04bf98284f01
# ╟─960dc301-0e7b-4908-bbc1-10cb0190c3f2
# ╠═049272e7-c8cc-4fc3-9fdc-fc992aaab814
# ╟─4f1eff47-8430-462c-822a-73bce9d1a186
# ╠═562dbff1-c0c7-4d13-92ce-8274d0da315f
# ╠═c8073f02-3e1f-4417-b330-3c720c33e31d
# ╠═cf98008f-47e9-455b-b124-f235a89912db
# ╠═f828c4d4-5938-4b8b-ae0e-b7bcb3f3afe6
# ╠═958d4c35-bcfb-4443-99ee-720b75399a57
# ╠═32bf9f83-a45a-49d5-9d20-5d46ec6f4288
# ╠═b837eb3b-5819-49aa-b503-f6c4eb6b153c
# ╠═a3bcd304-03d8-46fc-bc7a-cc8e5f5bb120
# ╠═661785d2-9302-45b9-9b86-d74903ec895a
# ╠═45440a46-a3ec-43e7-8033-b88dbb4776ca
# ╟─59a9e17d-e0a9-4a43-a82b-5b7884b6d2f4
# ╟─e36b3ba3-e922-499c-a87d-2895ced1e6f3
# ╟─c2eada85-41c4-46a1-ad52-a39230a73ca7
# ╠═e4866712-03b7-4e9f-b52b-c169def8533e
