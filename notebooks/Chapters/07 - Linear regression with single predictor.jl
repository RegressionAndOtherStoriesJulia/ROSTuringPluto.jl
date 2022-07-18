### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 2b08cf3d-a148-4981-a389-2abfdc622bf7
using Pkg, DrWatson

# ╔═╡ a28841db-e3f9-4c90-8b7b-144814616800
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

# ╔═╡ f39782c5-a0d0-4546-9e0d-bd0ff6eeaa95
md"## See chapter 7 in Regression and Other Stories."

# ╔═╡ 92ce35e9-ac0a-4f56-a4f8-0649545f4fcf
md" ##### Widen the notebook."

# ╔═╡ ac149089-83e8-45f3-9f64-e55fcab01c5f
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

# ╔═╡ 7cb5adb7-fefd-44b5-b60e-37eafdd1e6a0
md"##### A typical set of Julia packages to include in notebooks."

# ╔═╡ a8d7b228-2f7a-434b-b56d-630ae574e233
md"### 7.1 Example: Predicting presidential vote from the economy."

# ╔═╡ ec89f6e8-0e7b-4be4-9e9d-eec9a800e361
hdi = CSV.read(ros_datadir("HDI", "hdi.csv"), DataFrame)

# ╔═╡ eb645d53-9b17-4361-8d21-8d177c0f3393
hibbs = CSV.read(ros_datadir("ElectionsEconomy", "hibbs.csv"), DataFrame)

# ╔═╡ 2c099d92-5e66-42ae-a00e-e80b3ad540b2
hibbs_lm = lm(@formula(vote ~ growth), hibbs)

# ╔═╡ 8be7f5f7-8ccd-48ce-a097-265ad080eb64
residuals(hibbs_lm)

# ╔═╡ 9481b5e3-6128-4453-8e9f-47a7348046b9
mad(residuals(hibbs_lm))

# ╔═╡ 077cad7f-a2dd-47ef-90dc-96e73819be96
std(residuals(hibbs_lm))

# ╔═╡ 896816d9-6885-44d8-88a6-e4479e39068e
coef(hibbs_lm)

# ╔═╡ 0889d614-d08a-4a74-aefd-d60fb6f6cb03
let
	fig = Figure()
	hibbs.label = string.(hibbs.year)
	xlabel = "Average growth personal income [%]"
	ylabel = "Incumbent's party vote share"
	let
		title = "Forecasting the election from the economy"
		plt = data(hibbs) * 
			mapping(:label => verbatim, (:growth, :vote) => Point) *
			visual(Annotations, textsize=15)
		axis = (; title, xlabel, ylabel)
		draw!(fig[1, 1], plt; axis)
	end
	let
		title = "Data and linear fit"
		cols = mapping(:growth, :vote)
		scat = visual(Scatter) + linear()
		plt = data(hibbs) * cols * scat
		axis = (; title, xlabel, ylabel)
		draw!(fig[1, 2], plt; axis)
		annotations!("vote = 46.2 + 3.0 * growth"; position=(0, 41))
	end
	fig
end

# ╔═╡ def495f6-3124-4d42-9d3a-4f09a137cc6c
@model function ppl7_1(growth, vote)
    a ~ Normal(50, 20)
    b ~ Normal(2, 10)
    σ ~ Exponential(1)
    μ = a .+ b .* growth
    for i in eachindex(vote)
        vote[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ b4a66056-959e-4c9d-9252-13eedc69d89e
begin
	m7_1t = ppl7_1(hibbs.growth, hibbs.vote)
	chns7_1t = sample(m7_1t, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns7_1t)
end

# ╔═╡ 160957b0-3ee7-452d-a725-014ec9c09a07
begin
	post7_1t = DataFrame(chns7_1t)[:, 3:5]
	ms7_1t = model_summary(post7_1t, names(post7_1t))
end

# ╔═╡ 85766eab-6a2e-488f-a3a7-294465bd81c8
trankplot(post7_1t, "b")

# ╔═╡ 50ce2c67-6b33-4c22-afdc-0459154bd64e
let
	growth_range = LinRange(minimum(hibbs.growth), maximum(hibbs.growth), 200)
	votes = mean.(link(post7_1t, (r,x) -> r.a + x * r.b, growth_range))

	fig = Figure()
	xlabel = "Average growth personal income [%]"
	ylabel="Incumbent's party vote share"
	ax = Axis(fig[1, 1]; title="Regression line based on 4000 posterior samples", 
		subtitle = "(grey lines are based on first 200 draws of :a and :b)",
		xlabel, ylabel)
	for i in 1:200
		lines!(growth_range, post7_1t.a[i] .+ post7_1t.b[i] .* growth_range, color = :lightgrey)
	end
	scatter!(hibbs.growth, hibbs.vote)
	lines!(growth_range, votes, color = :red)
	fig
end

# ╔═╡ 2559d7d6-400c-44ed-afd6-6572ce50ac44
let
	println(46.3 + 3 * 2.0) # 52.3, σ = 3.6 (from ms7_1s above)
	probability_of_Clinton_winning = 1 - cdf(Normal(52.3, 3.6), 50)
end

# ╔═╡ cae9dea3-010e-48f9-b45f-1230b4456031
let
	f = Figure()
	ax = Axis(f[1, 1]; title = "")
	x_range = LinRange(30, 70, 100)
	y = pdf.(Normal(52.3, 3.6), x_range)
	lines!(x_range, y)

	x1 = range(50, 70; length=200)
	band!(x1, fill(0, length(x1)), pdf.(Normal(52.3, 3.6), x1);
		color = (:grey, 0.75), label = "Label")

	annotations!("Predicted\n74% change\nof Clinton victory", position=(51, 0.02), textsize=13)
	f
end

# ╔═╡ 792dbc25-9c7f-4bea-8f67-830d1d9369d4
md" ### 7.2 Checking the model-fitting procedure using simulation."

# ╔═╡ ff398b3a-1689-458c-bb75-43ef649a4561
let
	a = 46.3
	b = 3.0
	sigma = 3.9
	x = hibbs.growth
	n = length(x)

	y = a .+ b .* x + rand(Normal(0, sigma), n)
	fake = DataFrame(x = x, y = y)

	data = (N=nrow(fake), vote=fake.y, growth=fake.x)
	global m7_2t = ppl7_1(fake.x, fake.y)
	global chns7_2t = sample(m7_2t, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns7_2t)
end

# ╔═╡ cf4e7277-514a-4338-b1b3-203399627701
begin
	post7_2t = DataFrame(chns7_2t)[:, 3:5]
	ms7_2t = model_summary(post7_2t, names(post7_2t))
end

# ╔═╡ f4d38b34-e3b8-454b-b4cd-0c381c5fa397
function sim(ppl)
	a = 46.3
	b = 3.0
	sigma = 3.9
	x = hibbs.growth
	n = length(x)

	y = a .+ b .* x + rand(Normal(0, sigma), n)
	#println(mean(y))
	m7_2t = ppl(x, y)
	chns7_2t = sample(m7_2t, NUTS(), MCMCThreads(), 1000, 4)
	post7_2t = DataFrame(chns7_2t)[:, 3:5]
	ms = model_summary(post7_2t, Symbol.([:a, :b, :sigma]))
	b̂ = ms[:b, :mean] 
	b_se = ms[:b, :std]

	(
		b̂ = b̂, 
		b_se = b_se,
		cover_68 = Int(abs(b - b̂) < b_se),
		cover_95 = Int(abs(b - b̂) < 2b_se)
	)
end

# ╔═╡ b3787442-f109-42c5-a1a5-ec199054d3ff
sim(ppl7_1)

# ╔═╡ 21dab61a-148d-4f99-ab7a-d006aa7bea62
# ╠═╡ show_logs = false
let
	n_fake = 100 # 1000
	df = DataFrame()
	cover_68 = Float64[]
	cover_95 = Float64[]

	for i in 1:n_fake
		res = sim(ppl7_1)
		append!(df, DataFrame(;res...))
	end
	describe(df)
end

# ╔═╡ 8ff83066-3a60-46bb-8667-9cad0a7f9309
md"
!!! note

In above cell, I have hidden the logs. To show them, click on the little circle with 3 dots."

# ╔═╡ e2b8951a-59df-4066-b39e-aba71e3141d5
md" ### 7.3 Formulating comparisons as regression models."

# ╔═╡ 21cfb523-a5be-4db9-a03b-0a5d149bff53
 begin
 	r₀ = [-0.3, 4.1, -4.9, 3.3, 6.4, 7.2, 10.7, -4.6, 4.7, 6.0, 1.1, -6.7, 10.2, 9.7, 5.6,
		1.7, 1.3, 6.2, -2.1, 6.5]
	[mean(r₀), std(r₀)/sqrt(length(r₀))]
 end

# ╔═╡ 8498471e-17d5-424a-aa83-822b59fd097c
begin
	Random.seed!(3)
	n₀ = 20
	y₀ = r₀
	fake_0 = DataFrame(y₀ = r₀)
	data_0 = (N = nrow(fake_0), y = fake_0.y₀)

	n₁ = 30
	y₁ = rand(Normal(8.0, 5.0), n₁)
	data_1 = (N = n₁, y = y₁)

	se_0 = std(y₀)/sqrt(n₀)
	se_1 = std(y₁)/sqrt(n₁)
	
	(diff=mean(y₁)-mean(y₀), se_0=se_0, se_1=se_1, se=sqrt(se_0^2 + se_1^2))
end

# ╔═╡ 9a6403c9-f744-4dab-a918-c5e72b0ea341
@model function ppl7_3(y)
    a ~ Uniform(0, 10)
    σ ~ Uniform(0, 10)
    y ~ Normal(a, σ)
end

# ╔═╡ 14dec02d-8ff3-4b78-8fab-6af8126890fa
begin
	m7_3at = ppl7_3(data_0.y)
	chns7_3at = sample(m7_3at, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns7_3at)
end

# ╔═╡ e4b043af-68a7-455b-9bdb-d2d46839d7d3
begin
	post7_3at = DataFrame(chns7_3at)[:, 3:4]
	ms7_3at = model_summary(post7_3at, names(post7_3at))
end

# ╔═╡ bc5c69d8-6c90-4af4-96f2-d4fb4c7dd9eb
begin
	m7_3bt = ppl7_3(data_1.y)
	chns7_3bt = sample(m7_3bt, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns7_3bt)
end

# ╔═╡ 39ab4991-3df0-4d90-86cf-d595b2d869a6
begin
	post7_3bt = DataFrame(chns7_3bt)[:, 3:4]
	ms7_3bt = model_summary(post7_3bt, names(post7_3bt))
end

# ╔═╡ c12df31c-d9cd-4d8a-96a7-b5e10c339428
@model function ppl7_3c(x, y)
    a ~ Normal()
	b ~ Normal()
    σ ~ Exponential(1)
    μ = a .+ b .* x
    for i in eachindex(y)
        y[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ f364f536-3b00-4892-a0c4-3eaf07bafccd
# ╠═╡ show_logs = false
let
	n = n₀ + n₁
	y = vcat(y₀, y₁)
	x = vcat(zeros(Int, n₀), ones(Int, n₁))
	global fake = DataFrame(x=x, y=y)
	global m7_3ct = ppl7_3c(fake.x, fake.y)
	global chns7_3ct = sample(m7_3ct, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns7_3ct)
end

# ╔═╡ 8e4cb62d-8252-4203-a345-00fb8628d9bb
begin
	post7_3ct = DataFrame(chns7_3ct)[:, 3:6]
	sm7_3ct = model_summary(post7_3ct, [:a, :b, :σ])
end

# ╔═╡ a4ffdd25-d2f7-43e4-83a7-176d813f9d48
let
	f = Figure()
	ax = Axis(f[1, 1]; title="Least-squares regression on an indicator is\nthe same as computing a difference in means",
	xlabel="Indicator, x", ylabel="y")
	x_range = LinRange(0, 1, 100)
	â, b̂, σ̂ = sm7_3ct[:, :median]

	y = â .+ b̂ .* x_range
	lines!(x_range, y)
	x = vcat(zeros(Int, n₀), ones(Int, n₁))
	scatter!(fake.x, fake.y)
	ȳ₀ = mean(y₀)
	ȳ₁ = mean(y₁)
	hlines!(ax, [ȳ₀, ȳ₁]; color=:lightgrey)
	annotations!("ȳ₀ = $(round(ȳ₀, digits=1))", position=(0.05, 2.4), textsize=15)
	annotations!("ȳ₁ = $(round(ȳ₁, digits=1))", position=(0.9, 8.2), textsize=15)
	annotations!("y = $(round(â, digits=1)) + $(round(b̂, digits=1)) * x", position=(0.43, 4.4), textsize=15)
	f
end

# ╔═╡ fd091f91-cac2-44d8-92da-3cb2f470d832
mean(y₁)

# ╔═╡ Cell order:
# ╟─f39782c5-a0d0-4546-9e0d-bd0ff6eeaa95
# ╟─92ce35e9-ac0a-4f56-a4f8-0649545f4fcf
# ╠═ac149089-83e8-45f3-9f64-e55fcab01c5f
# ╠═2b08cf3d-a148-4981-a389-2abfdc622bf7
# ╟─7cb5adb7-fefd-44b5-b60e-37eafdd1e6a0
# ╠═a28841db-e3f9-4c90-8b7b-144814616800
# ╟─a8d7b228-2f7a-434b-b56d-630ae574e233
# ╠═ec89f6e8-0e7b-4be4-9e9d-eec9a800e361
# ╠═eb645d53-9b17-4361-8d21-8d177c0f3393
# ╠═2c099d92-5e66-42ae-a00e-e80b3ad540b2
# ╠═8be7f5f7-8ccd-48ce-a097-265ad080eb64
# ╠═9481b5e3-6128-4453-8e9f-47a7348046b9
# ╠═077cad7f-a2dd-47ef-90dc-96e73819be96
# ╠═896816d9-6885-44d8-88a6-e4479e39068e
# ╠═0889d614-d08a-4a74-aefd-d60fb6f6cb03
# ╠═def495f6-3124-4d42-9d3a-4f09a137cc6c
# ╠═b4a66056-959e-4c9d-9252-13eedc69d89e
# ╠═160957b0-3ee7-452d-a725-014ec9c09a07
# ╠═85766eab-6a2e-488f-a3a7-294465bd81c8
# ╠═50ce2c67-6b33-4c22-afdc-0459154bd64e
# ╠═2559d7d6-400c-44ed-afd6-6572ce50ac44
# ╠═cae9dea3-010e-48f9-b45f-1230b4456031
# ╟─792dbc25-9c7f-4bea-8f67-830d1d9369d4
# ╠═ff398b3a-1689-458c-bb75-43ef649a4561
# ╠═cf4e7277-514a-4338-b1b3-203399627701
# ╠═f4d38b34-e3b8-454b-b4cd-0c381c5fa397
# ╠═b3787442-f109-42c5-a1a5-ec199054d3ff
# ╠═21dab61a-148d-4f99-ab7a-d006aa7bea62
# ╟─8ff83066-3a60-46bb-8667-9cad0a7f9309
# ╟─e2b8951a-59df-4066-b39e-aba71e3141d5
# ╠═21cfb523-a5be-4db9-a03b-0a5d149bff53
# ╠═8498471e-17d5-424a-aa83-822b59fd097c
# ╠═9a6403c9-f744-4dab-a918-c5e72b0ea341
# ╠═14dec02d-8ff3-4b78-8fab-6af8126890fa
# ╠═e4b043af-68a7-455b-9bdb-d2d46839d7d3
# ╠═bc5c69d8-6c90-4af4-96f2-d4fb4c7dd9eb
# ╠═39ab4991-3df0-4d90-86cf-d595b2d869a6
# ╠═c12df31c-d9cd-4d8a-96a7-b5e10c339428
# ╠═f364f536-3b00-4892-a0c4-3eaf07bafccd
# ╠═8e4cb62d-8252-4203-a345-00fb8628d9bb
# ╠═a4ffdd25-d2f7-43e4-83a7-176d813f9d48
# ╠═fd091f91-cac2-44d8-92da-3cb2f470d832
