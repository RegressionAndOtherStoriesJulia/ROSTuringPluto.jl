### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 3ec83660-c767-49a4-ab27-923ea3443c98
using Pkg, DrWatson

# ╔═╡ 44233635-6129-4ccd-8bad-bbafa5287112
# ╠═╡ show_logs = false
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

	# Include basic packages
	using RegressionAndOtherStories
	import RegressionAndOtherStories: link
	
	# Common data files and functions
	using RegressionAndOtherStories

	set_aog_theme!()
	Logging.disable_logging(Logging.Warn)
end;

# ╔═╡ c87e36ac-2f7b-474c-9ad6-72c65f4d8e21
md" ## See chapter 3 in Regression and Other Stories."

# ╔═╡ 1b86b8b4-6892-4f41-ab89-fa3d1b3b7c0a
md" #### Widen the notebook"

# ╔═╡ e6974166-80e9-4903-b4bd-3f4cc3651823
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

# ╔═╡ d09f8d14-7c72-11ec-1481-976ff33e65af
md" ### 3.1 - Weighted averages"

# ╔═╡ af4f1bd4-ad6c-4f9f-a719-29ab5579b001
pop = DataFrame(stratum=1:3, country=["United States", "Mexico", "Canada"],
	population=Int[310e6, 112e6, 34e6], average_age=[36.8, 26.7, 40.7])

# ╔═╡ 900d5990-a4a6-4213-8965-86bf723050b6
mean(pop.average_age, weights(pop.population))

# ╔═╡ bed5f962-d1c0-4ce0-a158-665227846c0b
weights(pop.population)/sum(pop.population)

# ╔═╡ 2696411e-44fe-4fd9-8235-60c90ff61abf
describe(pop)

# ╔═╡ e4257130-e617-4c66-81a9-55f7c2d7e721
md" ### 3.3 - Graphing a line"

# ╔═╡ a2533602-96ea-4e2a-824b-9ae6443cf718
let
	data = LinRange(0.01, 0.99, 200)
	f = Figure(resolution = (800, 800))
	
	for (i, scale) in enumerate([identity, log10, log2, log, sqrt, Makie.logit])
	
	    row, col = fldmod1(i, 2)
	    Axis(f[row, col], yscale = scale, title = string(scale),
	        yminorticksvisible = true, yminorgridvisible = true,
	        yminorticks = IntervalsBetween(8))
	
	    lines!(data, color = :blue)
	end
	
	f
end

# ╔═╡ 3de3d4b5-a946-415b-9a0d-ef6d115d695f
let
	data = 10 .^ LinRange(0.01, 5.0, 200)
	f = Figure(resolution = (800, 300))
	
	for (i, scale) in enumerate([log10, log2, log])
	
	    row, col = fldmod1(i, 2)
	    Axis(f[1, i], yscale = scale, xscale = scale, title = string(scale),
	        yminorticksvisible = true, yminorgridvisible = true,
	        yminorticks = IntervalsBetween(8))
	
	    lines!(data, color = :blue)
	end
	
	f
end

# ╔═╡ acde5683-93e7-43b5-b5bc-1d5c8b10f3b7
let
	f = Figure()
	ax = Axis(f[1, 1]; title = "Mile record", subtitle = "(from 1900 to 2000)", xlabel = "Year", ylabel = "Time [sec]")
	xlims!(ax, 1900, 2000)
	ax.xticks = 1900:25:2000
	x = 1900:2000
	y = 1007 .- 0.393 .* x
	lines!(x, y)
	annotations!("y = 1007 - 0.393 * x", position=(1950, 250))
	current_figure()
end

# ╔═╡ f4b754ab-a772-47ac-bc28-aa0772037657
md" ### 3.4 - Log and exponential scales"

# ╔═╡ a25ab279-3358-4030-95e9-6e4a40a9ccd6
md" #### Simulated data for metabolic."

# ╔═╡ 4fffa589-79ee-48e5-9a2a-eba3bd361082
begin
	x = sort(rand(Uniform(0.01, 10000), 200))
	y = 4.1 * x.^0.74 .+ [rand.(Normal.(0, sqrt(x[i])), 1)[1] for i in 1:length(x)]
	metabolic = DataFrame(:body_mass => log.(x), :rate => log.(y))
end

# ╔═╡ 99eeda55-dcd3-46cf-bad1-4b7b65621139
@model function ppl3_1(m, r)
    a ~ Normal(0, 0.3)
    b ~ Normal(0, 0.3)
    σ ~ Exponential(1)
    μ = a .+ b .* m
    for i in eachindex(y)
        r[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ f038c7c3-0553-4130-b8ea-08a824bd0ee0
begin
	m3_1t = ppl3_1(metabolic.body_mass, metabolic.rate)
	chns3_1t = sample(m3_1t, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns3_1t)
end

# ╔═╡ c7f02711-aae1-43f2-ae05-b26d1ce9e18f
describe(chns3_1t)

# ╔═╡ b0ae1ed5-ba19-4c03-bdfb-4792b4eff2d4
begin
	post3_1t = DataFrame(chns3_1t)[:, 3:5]
	ms3_1t = model_summary(post3_1t, names(post3_1t))
end

# ╔═╡ 5a34151a-9bdd-4401-9dba-71be5bda01ba
let
	x = LinRange(1, 10000, 1000)
	y = 4.1 * x.^0.74
	f = Figure()
	ax = Axis(f[1, 1]; title="Metabolic rate (linear scale)", xlabel="Body mass [kg]", ylabel="Metobolic rate [W]")
	scatter!(exp.(metabolic.body_mass), exp.(metabolic.rate))
	lines!(x, y; color=:darkred)

	ax = Axis(f[1 , 2]; title="Metabolic rate (log-log scale)", xscale=log, yscale=log,
		xlabel="log(body mass [kg])", ylabel="log(metobolic rate [W])")
	x = LinRange(minimum(metabolic.body_mass), maximum(metabolic.body_mass), 100)
	scatter!(metabolic.body_mass, metabolic.rate)
	lines!(x, ms3_1t["a", "median"] .+ ms3_1t["b", "median"] * x; color=:darkred)
	current_figure()
end

# ╔═╡ 9efe7db8-c71c-4e2d-ad5d-33c90338373a
typeof(ms3_1t)

# ╔═╡ af373f88-7500-4c82-adbb-b0eb55c51b74
LinRange(minimum(metabolic.body_mass), maximum(metabolic.body_mass), 100)

# ╔═╡ b2b7e78f-3425-43cc-9795-d881667627cb
exp(exp(1.4))

# ╔═╡ fdf62f24-f0f4-46e1-b11a-cceb959bc4e8
md" ### 3.5 - Probability distributions"

# ╔═╡ 01d8d8b0-82cc-4067-af1e-ca72062e72a2
begin
	N = 100000
	heights = DataFrame()
	height = vcat(rand(Normal(63.7, 2.7), N), 
		rand(Normal(69.1, 2.9), N))
	sex = repeat(["female", "male"], inner=N)
	heights.height = height
	heights.sex = sex
	heights
end

# ╔═╡ 382060b9-7a6f-48df-871a-75fe5f163556
begin
	menHeights = heights[heights.sex .== "male", :height]
	womenHeights = heights[heights.sex .== "female", :height]
	(mean=mean(womenHeights), var=var(womenHeights), 
		std=std(womenHeights), median=median(womenHeights), 
		mad_sd=mad(womenHeights))
end

# ╔═╡ 65d3b481-8c53-4bb2-a9b8-12d9afdfe6c4
let
		f = Figure()
		ax = Axis(f[1, 1]; title="Density heights")
		density!(womenHeights; color=color = (:lightgreen, 0.4), label="Women")
		density!(menHeights; color=color = (:lightblue, 0.4), label="Men")
		axislegend()
		f
	end

# ╔═╡ 4066ca52-f609-4d7b-aa08-42210ff7ec7e
begin
	wdf = Normal(63.65, 2.68)
	cdf(wdf, 63.65 + 0.67 * 2.68) - cdf(wdf, 63.65 - 0.67 * 2.68)
end

# ╔═╡ cd48724d-5a75-4eb1-a5b3-860aee4f16a9
cdf(wdf, 63.65 + 2.68) - cdf(wdf, 63.65 - 2.68)

# ╔═╡ 72b07373-b294-486d-83a7-9c34b56c85c2
cdf(wdf, 63.65+2*2.68) - cdf(wdf, 63.65-2*2.68)

# ╔═╡ 2271ca22-6d77-4d5c-b501-d8aa21abc66d
let
	wdf = Normal(63.65, 2.68)
	x = range(55.0, 72.5 ; length=100)
	lines(x, pdf.(wdf, x); color=:darkblue)
	
	x1 = range(63.65 - 3 * 2.68, 63.65 - 2 * 2.68; length=20)
	band!(x1, fill(0, length(x1)), pdf.(wdf, x1);
		color = (:blue, 0.25), label = "Label")

	x1 = range(63.65 + 2 * 2.68, 63.65 + 3 * 2.68; length=20)
	band!(x1, fill(0, length(x1)), pdf.(wdf, x1);
		color = (:blue, 0.25), label = "Label")
	
	x1 = range(63.65 - 2 * 2.68, 63.65 - 1 * 2.68; length=20)
	band!(x1, fill(0, length(x1)), pdf.(wdf, x1);
		color = (:blue, 0.45), label = "Label")

	x1 = range(63.65 + 1 * 2.68, 63.65 + 2 * 2.68; length=20)
	band!(x1, fill(0, length(x1)), pdf.(wdf, x1);
		color = (:blue, 0.45), label = "Label")
	
	x1 = range(63.65 - 1 * 2.68, 63.65; length=20)
	band!(x1, fill(0, length(x1)), pdf.(wdf, x1);
		color = (:blue, 0.55), label = "Label")

	x1 = range(63.65, 63.65 + 2.68; length=20)
	band!(x1, fill(0, length(x1)), pdf.(wdf, x1);
		color = (:blue, 0.55), label = "Label")

	text!("68%", position = (63.65, 0.05), align = (:center,  :center),
    	textsize = 30)
	text!("13.5%", position = (67.5, 0.02), align = (:center,  :center),
    	textsize = 20)
	text!("13.5%", position = (59.6, 0.02), align = (:center,  :center),
    	textsize = 20)
	text!("2.5%", position = (69.75, 0.0045), align = (:center,  :center),
    	textsize = 15)
	text!("2.5%", position = (57.7, 0.0045), align = (:center,  :center),
    	textsize = 15)
	current_figure()
end

# ╔═╡ 52e90f79-ae39-4446-a972-db84a7a1c351
let
	n = 20; p = 0.3
	y = rand(Binomial(n, p), 10000)
	(m̂ = mean(y), m = 20 * 0.3, σ̂ = std(y), σ = sqrt(n * p * (1 - p)))
end

# ╔═╡ 356b24fb-d060-4914-8b2b-5bf3572d3f2b
let
	n = 20; p = 0.3
	y = rand(Bernoulli(p), 10000)
	(m̂ = mean(y), m = 0.3)
end

# ╔═╡ 381b7b38-4f06-4758-832f-83f561da1287
md" #### LogNormal"

# ╔═╡ 5d77b830-5aec-40ab-8c7d-6800c6c635cd
let
	menw = rand(LogNormal(5.13, 0.17), 10000)
	menwl = log.(menw)
	f = Figure()
	ax = Axis(f[1, 1]; title="Log weights of men\n(Normal distribution)", xlabel="Logarithm of weight [lb]", ylabel="Density")
	density!(menwl)
	ax = Axis(f[1, 2]; title="Weights of men\n(LogNormal distribution", xlabel="Weight of men [lb]", ylabel="Density")
	density!(menw)
	current_figure()
end

# ╔═╡ 34b8a308-1968-4742-8c3a-2d1296c8a12e
md" #### Binomial"

# ╔═╡ 954d0dc5-e673-4cc1-8b8d-4ac3d21ce70d
let
	df = DataFrame(bv = rand(Binomial(20, 0.3), 1000))
	model_summary(df, [:bv])
end

# ╔═╡ 45e693b1-06f1-4f2d-a835-8d0cedb8d998
begin
	n = 20
	p = 0.3 
	(mean = n * p, std = √(n * p * (1 - p)))
end

# ╔═╡ cb80aa2f-dbc3-4986-b48d-1cab66b1c97d
md" #### Poisson"

# ╔═╡ 40fde7c9-19b8-4185-b9c7-fd827b46ce3b
rand(Poisson(4.52), 10)

# ╔═╡ c4007f18-9428-4209-a0f0-2123f2c14217
md" ### 3.6 - Probability modeling"

# ╔═╡ 8b27d512-46e2-42a9-84aa-9a8fe37a2d7f
1 / (pdf(Normal(0.49, 0.04), 0.5) / 200000)

# ╔═╡ cd32ebf0-2568-4960-93f2-aacaaf47cee0
1 / (1000pdf(Normal(0.49, 0.04), 0.5) / 200000)

# ╔═╡ Cell order:
# ╟─c87e36ac-2f7b-474c-9ad6-72c65f4d8e21
# ╟─1b86b8b4-6892-4f41-ab89-fa3d1b3b7c0a
# ╠═e6974166-80e9-4903-b4bd-3f4cc3651823
# ╠═3ec83660-c767-49a4-ab27-923ea3443c98
# ╠═44233635-6129-4ccd-8bad-bbafa5287112
# ╟─d09f8d14-7c72-11ec-1481-976ff33e65af
# ╠═af4f1bd4-ad6c-4f9f-a719-29ab5579b001
# ╠═900d5990-a4a6-4213-8965-86bf723050b6
# ╠═bed5f962-d1c0-4ce0-a158-665227846c0b
# ╠═2696411e-44fe-4fd9-8235-60c90ff61abf
# ╟─e4257130-e617-4c66-81a9-55f7c2d7e721
# ╠═a2533602-96ea-4e2a-824b-9ae6443cf718
# ╠═3de3d4b5-a946-415b-9a0d-ef6d115d695f
# ╠═acde5683-93e7-43b5-b5bc-1d5c8b10f3b7
# ╟─f4b754ab-a772-47ac-bc28-aa0772037657
# ╟─a25ab279-3358-4030-95e9-6e4a40a9ccd6
# ╠═4fffa589-79ee-48e5-9a2a-eba3bd361082
# ╠═99eeda55-dcd3-46cf-bad1-4b7b65621139
# ╠═f038c7c3-0553-4130-b8ea-08a824bd0ee0
# ╠═c7f02711-aae1-43f2-ae05-b26d1ce9e18f
# ╠═b0ae1ed5-ba19-4c03-bdfb-4792b4eff2d4
# ╠═5a34151a-9bdd-4401-9dba-71be5bda01ba
# ╠═9efe7db8-c71c-4e2d-ad5d-33c90338373a
# ╠═af373f88-7500-4c82-adbb-b0eb55c51b74
# ╠═b2b7e78f-3425-43cc-9795-d881667627cb
# ╟─fdf62f24-f0f4-46e1-b11a-cceb959bc4e8
# ╠═01d8d8b0-82cc-4067-af1e-ca72062e72a2
# ╠═382060b9-7a6f-48df-871a-75fe5f163556
# ╠═65d3b481-8c53-4bb2-a9b8-12d9afdfe6c4
# ╠═4066ca52-f609-4d7b-aa08-42210ff7ec7e
# ╠═cd48724d-5a75-4eb1-a5b3-860aee4f16a9
# ╠═72b07373-b294-486d-83a7-9c34b56c85c2
# ╠═2271ca22-6d77-4d5c-b501-d8aa21abc66d
# ╠═52e90f79-ae39-4446-a972-db84a7a1c351
# ╠═356b24fb-d060-4914-8b2b-5bf3572d3f2b
# ╟─381b7b38-4f06-4758-832f-83f561da1287
# ╠═5d77b830-5aec-40ab-8c7d-6800c6c635cd
# ╟─34b8a308-1968-4742-8c3a-2d1296c8a12e
# ╠═954d0dc5-e673-4cc1-8b8d-4ac3d21ce70d
# ╠═45e693b1-06f1-4f2d-a835-8d0cedb8d998
# ╟─cb80aa2f-dbc3-4986-b48d-1cab66b1c97d
# ╠═40fde7c9-19b8-4185-b9c7-fd827b46ce3b
# ╟─c4007f18-9428-4209-a0f0-2123f2c14217
# ╠═8b27d512-46e2-42a9-84aa-9a8fe37a2d7f
# ╠═cd32ebf0-2568-4960-93f2-aacaaf47cee0
