### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 8442b25c-8331-4b77-82d7-c0a1089478f2
using Pkg, DrWatson

# ╔═╡ 7d2229ba-483a-417b-9ef7-2bf81fd71944
begin
	# Specific to this notebook
    using GLM

	# Specific to ROSTuringPluto
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

# ╔═╡ 6d0bb5ac-efc0-445a-aded-ec69fee95019
md"### See chapter 4 in Regression and Other Stories."

# ╔═╡ c0658351-3395-46a4-b711-98fd5988893f
md" ###### Widen the notebook."

# ╔═╡ 659d2b04-a03e-4faa-b818-1ae53b219f85

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

# ╔═╡ 169b41f7-7825-4abf-ae2b-e076e634b58c
md"###### A typical set of Julia packages to include in notebooks."

# ╔═╡ 9ead40d7-e641-4c93-bacd-bfe164b77417
md" #### 4.1 Sampling distributions and generative models."

# ╔═╡ 072d2912-363e-4d66-9f6f-241518b0349e
begin
	Random.seed!(1)
	a = 1.0
	b = 2.0
	x = LinRange(-2, 2, 100)
	y = a .+ b .* x .+ rand(Normal(0.0, 0.2), 100)
end;

# ╔═╡ ac86e192-f1f1-4722-85a6-89d95a8f7265
let
	f = Figure()
	ax = Axis(f[1, 1]; title="Linear regression")
	scatter!(x, y)
	lines!(x, a .+ b .* x; color=:darkred)
	annotations!("yᵢ = 1.0 + 2.0 * xᵢ + ϵᵢ", position=(0, -2), textsize=20)
	current_figure()
end

# ╔═╡ 0e763bc8-a53e-4c2d-b304-64cc338538bf
@model function ppl4_1(x, y)
    a ~ Normal(0, 1.5)
    b ~ Normal(1.0, 1.5)
    σ ~ Exponential(1)
    μ = a .+ b .* x
    for i in eachindex(y)
        y[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ c5a4783d-5763-4e1b-be3e-bdb560391833
begin
	m4_1t = ppl4_1(x, y)
	chns4_1t = sample(m4_1t, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns4_1t)
end

# ╔═╡ 71d89dae-4c6c-4002-b779-5ea0d5f17072
begin
	post4_1t = DataFrame(chns4_1t)[:, 3:5]
	ms4_1t = model_summary(post4_1t, names(post4_1t))
end

# ╔═╡ 4fd36f4b-28d1-47f2-a803-f8727e078484
let
	f = Figure()
	ax = Axis(f[1, 1]; title="Sampling distribution of a")
	density!(post4_1t.a)
	ax = Axis(f[1, 2]; title="Sampling distribution of b")
	density!(post4_1t.b)
	ax = Axis(f[1, 3]; title="Sampling distribution of sigma")
	density!(post4_1t.σ)
	current_figure()
end

# ╔═╡ bb7a65ba-0003-4d5c-abe1-19e6f6820f06
let
	f = Figure()
	ax = Axis(f[1, 1]; title="Linear regression")
	scatter!(x, y)
	lines!(x, ms4_1t["a", "median"] .+ ms4_1t["b", "median"] .* x; color=:darkred)
	mean_a = round(ms4_1t["a", "mean"]; digits=2)
	mean_b = round(ms4_1t["b", "mean"]; digits=2)
	mean_σ = round(ms4_1t["σ", "mean"]; digits=2)
	annotations!("y = $(mean_a) + $(mean_b) * x + $(mean_σ)", position=(0, -2), textsize=20)
	current_figure()
end

# ╔═╡ 8342a3bd-1ffb-470b-be32-d3d8db1eedc2
md" #### 4.2 Estimates, standard errors, and confidence intervals."

# ╔═╡ edb83342-1518-4938-afca-b9981bfe6c48
let
	f = Figure()
	ax = Axis(f[1, 1]; title="Sampling distribution of b (revisited)")
	b̂ = ms4_1t["b", "median"]
	σ̂ = ms4_1t["b", "std"]
	x = LinRange(b̂ - 4σ̂ , b̂ + 4σ̂, 100)
	y = pdf.(Normal(b̂, σ̂), x)
	ylims!(ax, [0, maximum(y) + 1.0])
	ax.xticks = b̂ - 3σ̂ : σ̂ : b̂ + 3σ̂
	ax.xtickformat = xs -> ["$(i) s.e." for i in -3:3]
	lines!(x, y)
	vlines!(ax, b̂; ymax=maximum(y)/(maximum(y) + 1.0), color=:grey)
	vlines!(ax, b̂-σ̂; ymax=pdf.(Normal(b̂, σ̂), b̂-σ̂)/(maximum(y) + 1.0), color=:grey)
	vlines!(ax, b̂+σ̂; ymax=pdf.(Normal(b̂, σ̂), b̂+σ̂)/(maximum(y) + 1.0), color=:grey)
	annotations!("b ±  1 s.e.", position=(b̂-0.008, 7.5), textsize=20)
	x1 = range(b̂ - σ̂ , b̂ + σ̂; length=60)
	band!(x1, fill(0, length(x1)), pdf.(Normal(b̂, σ̂), x1); color = (:blue, 0.25))
	f
end

# ╔═╡ 47c24a34-c8fb-4909-8de6-f46c36fbda56
let
	n = 100
	b = 2.0

	f = Figure()
	ax = Axis(f[1,1]; title="Simulation of confidence intervals", xlabel="Simulation",
		ylabel="Assumed 50% and 95% confidence intervals")

	x = 1:n
	y = [rand(Uniform(b - 2.1 , b + 2.1), 1)[1] for i in 1:n]

	# Assumed s.e. = 1.0
	lowerrors = fill(0.66, n)
	higherrors = fill(2, n)
	
	errorbars!(x, y, lowerrors, color = :red) # same low and high error
	errorbars!(x, y, higherrors, color = :grey) # same low and high error
	
	scatter!(x, y, markersize = 3, color = :black)
	hlines!(ax, [2])
	
	f
end

# ╔═╡ 0890feea-2432-4472-b36e-54a36b2c02f9
let
	n = 1000
	yes = 700
	no = n - yes
	est = yes/n
	se = sqrt(est * (1 - est)/n)
	
	(estimate = est, se = se, int_95 = est .+ quantile.(Normal(0, 1), [0.025, 0.975]) * se)
end

# ╔═╡ 3d2b0416-8510-463d-9ed0-a598085d9918
let
	y = [35, 34, 38, 35, 37]
	n = length(y)
	est = mean(y)
	se = std(y)/sqrt(n)
	int_50 = est .+ quantile.(TDist(n-1), [0.25, 0.75]) * se
	int_95 =  est .+ quantile.(TDist(n-1), [0.025, 0.975]) * se
	
	(estimate = est, se = se, int_50 = int_50, int_95 = int_95)
end

# ╔═╡ 4be6f3e8-1391-4f3a-9b13-0f8537b92a34
df_poll = CSV.read(ros_datadir("Death", "polls.csv"), DataFrame)

# ╔═╡ 886c48b5-c6ee-4a2e-b156-ba33217cc39d
let
	f = Figure()
	ax = Axis(f[1, 1]; title="Death penalty opinions", xlabel="Year", ylabel="Percentage support for the death penalty")
	scatter!(df_poll.year, df_poll.support .* 100)
	global err_lims = [100(sqrt(df_poll.support[i]*(1-df_poll.support[i])/1000)) for i in 1:nrow(df_poll)]
	errorbars!(df_poll.year, df_poll.support .* 100, err_lims, color = :red)
	f
end

# ╔═╡ a8d0258a-0b17-45c9-985a-420f0bec67e3
err_lims

# ╔═╡ 769e98f4-f771-4084-9805-c93eaf95f934
md" ### 4.3 Bias and unmodeled uncertaincy."

# ╔═╡ b810964a-deb9-41fd-963e-7e6b21005417
md" ### 4.4 Statistical significance, hypothesis testing, and statistical erros."

# ╔═╡ a6d4b0bd-acf2-45af-90c9-36c330439179
md" ### 4.5 Problems with the concept of statistical significance."

# ╔═╡ e96d3afa-6a33-49a4-bd11-8d4a248d4fe5
md" ### 4.6 Example of hypothesis testing: 55,000 residents need your help!"

# ╔═╡ 6f35948d-963c-43ce-85d2-9450b4416093
md" ### 4.7 Moving beyond hypothesis testing."

# ╔═╡ Cell order:
# ╟─6d0bb5ac-efc0-445a-aded-ec69fee95019
# ╟─c0658351-3395-46a4-b711-98fd5988893f
# ╠═659d2b04-a03e-4faa-b818-1ae53b219f85
# ╠═8442b25c-8331-4b77-82d7-c0a1089478f2
# ╟─169b41f7-7825-4abf-ae2b-e076e634b58c
# ╠═7d2229ba-483a-417b-9ef7-2bf81fd71944
# ╟─9ead40d7-e641-4c93-bacd-bfe164b77417
# ╠═072d2912-363e-4d66-9f6f-241518b0349e
# ╠═ac86e192-f1f1-4722-85a6-89d95a8f7265
# ╠═0e763bc8-a53e-4c2d-b304-64cc338538bf
# ╠═c5a4783d-5763-4e1b-be3e-bdb560391833
# ╠═71d89dae-4c6c-4002-b779-5ea0d5f17072
# ╠═4fd36f4b-28d1-47f2-a803-f8727e078484
# ╠═bb7a65ba-0003-4d5c-abe1-19e6f6820f06
# ╟─8342a3bd-1ffb-470b-be32-d3d8db1eedc2
# ╠═edb83342-1518-4938-afca-b9981bfe6c48
# ╠═47c24a34-c8fb-4909-8de6-f46c36fbda56
# ╠═0890feea-2432-4472-b36e-54a36b2c02f9
# ╠═3d2b0416-8510-463d-9ed0-a598085d9918
# ╠═4be6f3e8-1391-4f3a-9b13-0f8537b92a34
# ╠═886c48b5-c6ee-4a2e-b156-ba33217cc39d
# ╠═a8d0258a-0b17-45c9-985a-420f0bec67e3
# ╟─769e98f4-f771-4084-9805-c93eaf95f934
# ╟─b810964a-deb9-41fd-963e-7e6b21005417
# ╟─a6d4b0bd-acf2-45af-90c9-36c330439179
# ╟─e96d3afa-6a33-49a4-bd11-8d4a248d4fe5
# ╟─6f35948d-963c-43ce-85d2-9450b4416093
