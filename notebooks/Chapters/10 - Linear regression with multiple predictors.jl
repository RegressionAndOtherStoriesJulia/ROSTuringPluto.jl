### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ da0eb04e-9930-4f49-809b-3ba5fe16a59c
using Pkg, DrWatson

# ╔═╡ d7992c39-0617-42b5-b977-f04260d2bd03
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

# ╔═╡ f90c8b82-a52d-45a4-86eb-d2c7eb905692
md"### KidIQ: kidiq.csv"

# ╔═╡ 2daed834-e53d-4530-98bd-e80b5b99162f
md" ###### Widen the notebook."

# ╔═╡ d1c76cd1-2537-470e-a9d5-ebaa5b9dec7e
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


# ╔═╡ dfe95bfa-507a-4831-b3b5-f26b1d537112
kidiq = CSV.read(ros_datadir("KidIQ", "kidiq.csv"), DataFrame)

# ╔═╡ 4a4605a4-084a-4d64-a444-fa0fcf41ef91
let
	f = Figure()
	ax = Axis(f[1, 1]; title="KidIQ data: kid_score ~ mom_hs")
	scatter!(kidiq[kidiq.mom_hs .== 0, :mom_hs], kidiq[kidiq.mom_hs .== 0, :kid_score]; color=:red, markersize = 3)
	scatter!(kidiq[kidiq.mom_hs .== 1, :mom_hs], kidiq[kidiq.mom_hs .== 1, :kid_score]; color=:blue, markersize = 3)
	ax = Axis(f[1, 2]; title="KidIQ data: kid_score ~ mom_iq")
	scatter!(kidiq[kidiq.mom_hs .== 0, :mom_iq], kidiq[kidiq.mom_hs .== 0, :kid_score]; color=:red, markersize = 3)
	scatter!(kidiq[kidiq.mom_hs .== 1, :mom_iq], kidiq[kidiq.mom_hs .== 1, :kid_score]; color=:blue, markersize = 3)
	current_figure()
end

# ╔═╡ d8d8e6b4-6406-43ac-9472-a7afab027aef
stan10_1 = "
data {
	int N;
	vector[N] mom_hs;
	vector[N] kid_score;
}
parameters {
	real a;
	real b;
	real sigma;
}
model {
	vector[N] mu;
	a ~ normal(100, 10);
	b ~ normal(5, 10);
	mu = a + b * mom_hs;
	kid_score ~ normal(mu, sigma);
}
";

# ╔═╡ cf817e35-2098-4acd-a518-4904db52f495
@model function ppl10_1(x, y)
	a ~ Normal(100, 10)
	b ~ Normal(5, 10)
  	σ ~ Exponential(1)
	μ = a .+ b .* x
	for i in eachindex(y)
		y[i] ~ Normal(μ[i], σ)
	end
end

# ╔═╡ a9fcb0f0-5134-48a8-8d93-bfb8203e089e
begin
	m10_1t = ppl10_1(kidiq.mom_hs, kidiq.mom_iq)
	chns10_1t = sample(m10_1t, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns10_1t)
end

# ╔═╡ 9073f433-fbec-4cf7-8ef0-ba870854eb75
begin
	post10_1t = DataFrame(chns10_1t)[:, 3:5]
	ms10_1t = model_summary(post10_1t, [:a, :b, :σ])
end

# ╔═╡ b77b121b-38d3-485c-a9a2-c9ae87fe3423
let
	f = Figure()
	ax = Axis(f[1, 1]; title="KidIQ data: kid_score ~ mom_hs")
	scatter!(kidiq[kidiq.mom_hs .== 0, :mom_hs], kidiq[kidiq.mom_hs .== 0, :kid_score]; color=:red, markersize = 3)
	scatter!(kidiq[kidiq.mom_hs .== 1, :mom_hs], kidiq[kidiq.mom_hs .== 1, :kid_score]; color=:blue, markersize = 3)
	lines!([0.0, 1.0], [ms10_1t[:a, :median], ms10_1t[:a, :median] + ms10_1t[:b, :median]])
	current_figure()
end

# ╔═╡ e348f77e-708f-43ed-863e-795330637846
stan10_2 = "
data {
	int N;
	vector[N] mom_iq;
	vector[N] kid_score;
}
parameters {
	real a;
	real b;
	real sigma;
}
model {
	vector[N] mu;
	a ~ normal(25, 3);
	b ~ normal(1, 2);
	mu = a + b * mom_iq;
	kid_score ~ normal(mu, sigma);
}
";

# ╔═╡ 3e15c7db-6c76-47a9-af3f-66a2eb67a8aa
@model function ppl10_2(x, y)
	a ~ Normal(25, 3)
	b ~ Normal(1, 2)
  	σ ~ Exponential(1)
	μ = a .+ b .* x
	for i in eachindex(y)
		y[i] ~ Normal(μ[i], σ)
	end
end

# ╔═╡ 358093fe-3102-4c21-be0a-5d33569c13fd
begin
	m10_2t = ppl10_2(kidiq.mom_iq, kidiq.kid_score)
	chns10_2t = sample(m10_2t, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns10_2t)
end

# ╔═╡ f4ab1e3d-8f64-4415-a257-63e9d43c60db
begin
	post10_2t = DataFrame(chns10_2t)[:, 3:5]
	ms10_2t = model_summary(post10_2t, [:a, :b, :sigma])
end

# ╔═╡ 90f0236a-2c5b-4ea8-83cf-490037bf8c15
let
	f = Figure()
	ax = Axis(f[1, 1]; title="KidIQ data: kid_score ~ mom_iq")
	scatter!(kidiq[kidiq.mom_hs .== 0, :mom_iq], kidiq[kidiq.mom_hs .== 0, :kid_score]; color=:red, markersize = 3)
	scatter!(kidiq[kidiq.mom_hs .== 1, :mom_iq], kidiq[kidiq.mom_hs .== 1, :kid_score]; color=:blue, markersize = 3)
	x = LinRange(70.0, 140.0, 100)
	lines!(x, ms10_2t[:a, :median] .+ ms10_2t[:b, :median] .* x)
	current_figure()
end

# ╔═╡ 1204cbcb-a765-461e-a477-ae3d9913b1bf
stan10_3 = "
data {
	int N;
	vector[N] mom_hs;
	vector[N] mom_iq;
	vector[N] kid_score;
}
parameters {
	real a;
	real b;
	real c;
	real sigma;
}
model {
	vector[N] mu;
	a ~ normal(25, 2);
	b ~ normal(5, 2);
	c ~ normal(1, 2);
	mu = a + b * mom_hs + c * mom_iq;
	kid_score ~ normal(mu, sigma);
}
";

# ╔═╡ bcf8c679-0e35-42e7-b4dd-91ac8dfb5721
@model function ppl10_3(x, y, z)
	a ~ Normal(25, 2)
	b ~ Normal(5, 2)
	c ~ Normal(1, 2)
  	σ ~ Exponential(1)
	μ = a .+ b .* x .+ c .* y
	for i in eachindex(y)
		z[i] ~ Normal(μ[i], σ)
	end
end

# ╔═╡ b1c6c6a5-1784-438a-b930-49ce7aef80ab
begin
	m10_3t = ppl10_3(kidiq.mom_hs, kidiq.mom_iq, kidiq.kid_score)
	chns10_3t = sample(m10_3t, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns10_3t)
end

# ╔═╡ b6330c4f-8129-4bbd-aa39-3ddd00c062b5
begin
	post10_3t = DataFrame(chns10_3t)[:, 3:6]
	ms10_3t = model_summary(post10_3t, [:a, :b, :c, :σ])
end

# ╔═╡ 6014b70b-c10d-4b91-94f7-79dc291cc92b
let
	momnohs(x) = x == 0
	nohs = findall(momnohs, kidiq.mom_hs)

	momhs(x) = x == 1
	hs = findall(momhs, kidiq.mom_hs)
	
	f = Figure()
	ax = Axis(f[1, 1]; title="KidIQ data: kid_score ~ mom_hs + mom_iq")
	sca1 = scatter!(kidiq[kidiq.mom_hs .== 0, :mom_iq], kidiq[kidiq.mom_hs .== 0, :kid_score]; color=:red, markersize = 3)
	sca2 = scatter!(kidiq[kidiq.mom_hs .== 1, :mom_iq], kidiq[kidiq.mom_hs .== 1, :kid_score]; color=:blue, markersize = 3)
	x = sort(kidiq.mom_iq[nohs])
	lin1 =lines!(x, ms10_3t[:a, :median] .+ ms10_3t[:b, :median] .* kidiq.mom_hs[nohs] .+ ms10_3t[:c, :median] .* x; 
		color=:darkred)
	x = sort(kidiq.mom_iq[hs])
	lin2 =lines!(x, ms10_3t[:a, :median] .+ ms10_3t[:b, :median] .* kidiq.mom_hs[hs] .+ ms10_3t[:c, :median] .* x; 	
		color=:darkblue)
	Legend(f[1, 2],
    	[sca1, sca2, lin1, lin2],
    	["No high school", "High school", "No high school", "High School"])
	current_figure()
end

# ╔═╡ Cell order:
# ╟─f90c8b82-a52d-45a4-86eb-d2c7eb905692
# ╟─2daed834-e53d-4530-98bd-e80b5b99162f
# ╠═d1c76cd1-2537-470e-a9d5-ebaa5b9dec7e
# ╠═da0eb04e-9930-4f49-809b-3ba5fe16a59c
# ╠═d7992c39-0617-42b5-b977-f04260d2bd03
# ╠═dfe95bfa-507a-4831-b3b5-f26b1d537112
# ╠═4a4605a4-084a-4d64-a444-fa0fcf41ef91
# ╠═d8d8e6b4-6406-43ac-9472-a7afab027aef
# ╠═cf817e35-2098-4acd-a518-4904db52f495
# ╠═a9fcb0f0-5134-48a8-8d93-bfb8203e089e
# ╠═9073f433-fbec-4cf7-8ef0-ba870854eb75
# ╠═b77b121b-38d3-485c-a9a2-c9ae87fe3423
# ╠═e348f77e-708f-43ed-863e-795330637846
# ╠═3e15c7db-6c76-47a9-af3f-66a2eb67a8aa
# ╠═358093fe-3102-4c21-be0a-5d33569c13fd
# ╠═f4ab1e3d-8f64-4415-a257-63e9d43c60db
# ╠═90f0236a-2c5b-4ea8-83cf-490037bf8c15
# ╠═1204cbcb-a765-461e-a477-ae3d9913b1bf
# ╠═bcf8c679-0e35-42e7-b4dd-91ac8dfb5721
# ╠═b1c6c6a5-1784-438a-b930-49ce7aef80ab
# ╠═b6330c4f-8129-4bbd-aa39-3ddd00c062b5
# ╠═6014b70b-c10d-4b91-94f7-79dc291cc92b
