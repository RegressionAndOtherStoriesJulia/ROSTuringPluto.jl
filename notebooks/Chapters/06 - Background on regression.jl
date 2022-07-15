### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 1271ba57-93ff-4ef7-bfff-15c39a034b2c
using Pkg, DrWatson

# ╔═╡ 71cd8293-8c62-42b3-a33e-def5f7192160
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

# ╔═╡ a0fa3631-557c-4bb4-8862-cd21e24655e1
md"#### See Chapter 6 in Regression and Other Stories."

# ╔═╡ 5a68738d-8f8f-44b1-af3c-f3ceed14d82b
md" ##### Widen the notebook."

# ╔═╡ f96ef9cd-c3e2-4796-af8c-e5d94198fd6b
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

# ╔═╡ 5356fcee-dada-48dd-b422-bed4b5595470
md"##### A typical set of Julia packages to include in notebooks."

# ╔═╡ 8dba464a-3eb7-43e5-8324-85b938be6d51
md" ### 6.1 Regression models."

# ╔═╡ 326e0fc7-978b-42ae-b532-bdafb0a4948a
md"### 6.2 Fitting a simple regression to fake data."

# ╔═╡ 30832e8b-0642-447b-b281-207d7c4d73f4
let
	n = 20
	x = LinRange(1, n, 20)
	a = 0.2
	b = 0.3
	sigma = 0.5
	y = a .+ b .* x .+ rand(Normal(0, sigma), n)
	global fake = DataFrame(x=x, y=y)
end

# ╔═╡ bf9d7328-9b2f-4465-a7c6-e011ebd0c86e
@model function ppl6_1(x, y)
    a ~ Uniform(-2, 2)
    b ~ Uniform(-2, 2)
    σ ~ Uniform(0, 10)
    μ = a .+ b .* x
    for i in eachindex(y)
        y[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ c30cc031-16bd-4691-92e1-c89ceb5cc1ca
begin
	m6_1t = ppl6_1(fake.x, fake.y)
	chns6_1t = sample(m6_1t, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns6_1t)
end

# ╔═╡ 3c9c37de-7254-4fa8-94ba-ee74045a5a97
begin
	post6_1t = DataFrame(chns6_1t)[:, 3:5]
	ms6_1t = model_summary(post6_1t, names(post6_1t))
end

# ╔═╡ 08be202b-f618-4a7f-a48a-5618b19495f5
let
	f = Figure()
	
	ax = Axis(f[1, 1]; title="Regression of fake data.", xlabel="fake.x", ylabel="fake.y")
	scatter!(fake.x, fake.y)
	x = 1:0.01:20
	y = ms6_1t["a", "median"] .+  ms6_1t["b", "median"] .* x
	lines!(x, y)
	â, b̂, σ̂  = round.(ms6_1t[:, "median"]; digits=2)
	annotations!("y = $(â) + $(b̂) * x + ϵ"; position=(5, 0.8))
	
	ax = Axis(f[1, 2]; title="Regression of fake data.", subtitle="(using the link() function)",
		xlabel="fake.x", ylabel="fake.y")
	scatter!(fake.x, fake.y)
	xrange = LinRange(1, 20, 200)
	y = mean.(link(post6_1t, (r,x) -> r.a + x * r.b, xrange))
	lines!(xrange, y)
	annotations!("y = $(â) + $(b̂) * x + ϵ"; position=(5, 0.8))
	
	current_figure()
end

# ╔═╡ 9a2ec98b-6f45-4bc2-8ac0-5d6337cfb644
DataFrame(parameters = Symbol.(names(post6_1t)), simulated = [0.2, 0.3, 0.5], median = ms6_1t[:, "median"], mad_sd = ms6_1t[:, "mad_sd"])

# ╔═╡ bc877661-9aa0-425d-88e0-b82239c2552c
md" ### 6.3 Interpret coefficients as comparisons, not effects."

# ╔═╡ 6e6c65b3-f7ad-4a7f-91fd-f065e7bd7ffe
begin
	earnings = CSV.read(ros_datadir("Earnings", "earnings.csv"), DataFrame)
	earnings[:, [:earnk, :height, :male]]
end

# ╔═╡ 197b0a6a-7c1c-4aad-b9ba-5743c02169fc
describe(earnings[:, [:earnk, :height, :male]])

# ╔═╡ 696dcadc-7b10-4095-b165-2dcde9f89001
@model function ppl6_2(male, height, earnk)
    a ~ Normal()
    b ~ Normal()
	c ~ Normal()
    σ ~ Exponential(1)
    μ = a .+ b .* height .+ c .* male
    for i in eachindex(earnk)
        earnk[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ 103de138-886b-446d-8752-255f9db9d978
begin
	m6_2t = ppl6_2(earnings.male, earnings.height, earnings.earnk)
	chns6_2t = sample(m6_2t, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns6_2t)
end

# ╔═╡ 7c96df85-6981-45dc-a912-311e1c4fbad1
begin
	post6_2t = DataFrame(chns6_2t)[:, 3:6]
	ms6_2t = model_summary(post6_2t, names(post6_2t))
end

# ╔═╡ 9727012b-1b76-4cde-84ef-bdd7bfa4b025
let
	â, b̂, ĉ, σ̂ = round.(ms6_2t[:, "median"]; digits=2)

	fig = Figure()
	
	ax = Axis(fig[1, 1]; title="Earnings for males", subtitle="earnk = $(round(ĉ + â; digits=2)) + $(b̂) * mheight + ϵ")
	m = sort(earnings[earnings.male .== 1, [:height, :earnk]])
	scatter!(m.height, m.earnk)
	mheight_range = LinRange(minimum(m.height), maximum(m.height), 200)
	earnk = mean.(link(post6_2t, (r,x) -> r.c + r.a + x * r.b, mheight_range))

	lines!(mheight_range, earnk; color=:darkred)

	ax = Axis(fig[1, 2]; title="Earnings for females", subtitle="earnk = $(â) + $(b̂) * fheight + ϵ")
	f = sort(earnings[earnings.male .== 0, [:height, :earnk]])
	scatter!(f.height, f.earnk)
	fheight_range = LinRange(minimum(f.height), maximum(f.height), 200)
	earnk = mean.(link(post6_2t, (r,x) -> r.a + x * r.b, fheight_range))
	lines!(fheight_range, earnk; color=:darkred)

	fig
end	

# ╔═╡ ec3f566f-278c-4342-a365-db379b7d54aa
R2 = 1 - ms6_2t["σ", "mean"]^2 / std(earnings.earnk)^2

# ╔═╡ c752672e-56de-4498-a85d-e3dffdc254d6
md" ### 6.4 Historical origins of regression."

# ╔═╡ db99f612-77b4-4639-892b-389a20c0ae83
heights = CSV.read(ros_datadir("PearsonLee", "heights.csv"), DataFrame)

# ╔═╡ da420651-95a6-4fdc-ad2f-dbe2f550f672
@model function ppl6_3(m_height, d_height)
    a ~ Normal()
    b ~ Normal()
    σ ~ Exponential(1)
    μ = a .+ b .* m_height
    for i in eachindex(d_height)
        d_height[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ 821e8c88-627f-41c3-8128-3d3b859575e0
begin
	m6_3t = ppl6_3(heights.mother_height, heights.daughter_height)
	chns6_3t = sample(m6_3t, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns6_3t)
end

# ╔═╡ a5cd0b09-eff2-42c5-ab13-39e003099f32
begin
	post6_3t = DataFrame(chns6_3t)[:, 3:5]
	ms6_3t = model_summary(post6_3t, names(post6_3t))
end

# ╔═╡ 1e38322c-821f-45a5-aad9-d4c114ae37b7
let
	f = Figure()
	ax = Axis(f[1, 1]; title="Mothers' and daugthers' heights")
	xlims!(ax, 51, 74)
	scatter!(jitter.(heights.mother_height), jitter.(heights.daughter_height); markersize=3)
	x_range = LinRange(51, 74, 100)
	lines!(x_range, mean.(link(post6_3t, (r, x) -> r.a + r.b * x, x_range)); color=:darkred)
	scatter!([mean(heights.mother_height)], [mean(heights.daughter_height)]; markersize=20)
	f
end

# ╔═╡ 444766e8-86b9-496b-8b11-979b08fa842e
let
	f = Figure()
	ax = Axis(f[1, 1]; title="Mothers` and daughters' heights,\naverage of data, and fitted regression line",
		xlabel="Mother's height [in]", ylabel="Adult daugther's height [in]")
	scatter!(heights.mother_height, heights.daughter_height; markersize=5)
	xrange = LinRange(50, 72, 100)
	y = 30 .+ 0.54 .* xrange
	m̄ = mean(heights.mother_height)
	d̄ = mean(heights.daughter_height)
	scatter!([m̄], [d̄]; markersize=20, color=:gray)
	lines!(xrange, y)
	vlines!(ax, m̄; ymax=0.55, color=:grey)
	hlines!(ax, d̄; xmax=0.58, color=:grey)
	annotations!("y = 30 + 0.54 * mother's height", position=(49, 55), textsize=15)
	annotations!("or: y = 63.9 + 0.54 * (mother's height - 62.5)", position=(49, 54), textsize=15)
	f
end

# ╔═╡ ae86b798-1d42-4bf0-8bde-7d4c922a48fd
let
	f = Figure()
	ax = Axis(f[1, 1]; title="Mothers` and daughters' heights,\naverage of data, and fitted regression line",
		xlabel="Mother's height [in]", ylabel="Adult daugther's height [in]")
	scatter!(heights.mother_height, heights.daughter_height; markersize=5)
	xrange = LinRange(0, 72, 100)
	y = 30 .+ 0.54 .* xrange
	m̄ = mean(heights.mother_height)
	d̄ = mean(heights.daughter_height)
	scatter!([m̄], [d̄]; markersize=20, color=:gray)
	lines!(xrange, y)
	annotations!("y = 30 + 0.54 * mother's height", position=(20, 35), textsize=15)
	annotations!("or: y = 63.9 + 0.54 * (mother's height - 62.5)", position=(20, 33), textsize=15)
	f
end

# ╔═╡ 1e40f68d-49ea-4700-95b6-88f69e527036
@model function ppl6_4(m_height, d_height)
    a ~ Normal(25, 3)
    b ~ Normal(0, 0.5)
    σ ~ Exponential(1)
    μ = a .+ b .* m_height
    for i in eachindex(d_height)
        d_height[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ 285f5142-6cfb-4038-843a-d800ceefff50
begin
	m6_4t = ppl6_4(heights.mother_height, heights.daughter_height)
	chns6_4t = sample(m6_4t, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns6_4t)
end

# ╔═╡ b7390db2-ccf3-4c27-be14-01eaedef7574
begin
	post6_4t = DataFrame(chns6_4t)[:, 3:5]
	ms6_4t = model_summary(post6_4t, names(post6_4t))
end

# ╔═╡ 1a132ed9-3ff5-42d2-bb63-642958abc5ce
plot_chains(post6_4t, [:a, :b, :σ])

# ╔═╡ aa015e5a-88ca-4699-8f39-d0c54d8679e5
trankplot(post6_4t, "b")

# ╔═╡ 2bd08619-ccd4-4ade-8d40-955c5fcd3fc2
md" ###### Above trankplot and the low `ess` numbers a couple of cells earlier do not look healthy."

# ╔═╡ 4c04253e-cb54-4f46-817e-af76053eef16
md" ### 6.5 The paradox of regression to the mean."

# ╔═╡ aee6f37f-2a8a-451b-96e4-ec4eeb852b20
let
	n = 1000
	true_ability = rand(Normal(50, 10), n)
	noise_1 = rand(Normal(0, 10), n)
	noise_2 = rand(Normal(0, 10), n)
	midterm = true_ability + noise_1
	final = true_ability + noise_2
	global exams = DataFrame(midterm=midterm, final=final)
end

# ╔═╡ 93d4f920-7e1a-473b-badc-49cfdcc5f456
@model function ppl6_5(midterm, final)
    a ~ Normal()
    b ~ Normal()
    σ ~ Exponential(1)
    μ = a .+ b .* midterm
    for i in eachindex(final)
        final[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ f78d1b5e-d713-4d51-918d-005e7ca845af
begin
	m6_5t = ppl6_5(exams.midterm, exams.final)
	chns6_5t = sample(m6_5t, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns6_5t)
end

# ╔═╡ 958459f8-02be-4a84-a3a2-f5d144d21103
begin
	post6_5t = DataFrame(chns6_5t)[:, 3:5]
	ms6_5t = model_summary(post6_5t, names(post6_5t))
end

# ╔═╡ c0aeefbd-db6f-4359-80c1-9ff6ef5bb6f3
df_poll = CSV.read(ros_datadir("Death", "polls.csv"), DataFrame)

# ╔═╡ d3ae1909-f6a3-437a-9d19-5b4a6e6baab3
let
	f = Figure()
	ax = Axis(f[1, 1]; title="Death penalty opinions", xlabel="Year", ylabel="Percentage support for the death penalty")
	scatter!(df_poll.year, df_poll.support .* 100)
	err_lims = [100(sqrt(df_poll.support[i]*(1-df_poll.support[i])/1000)) for i in 1:nrow(df_poll)]
	errorbars!(df_poll.year, df_poll.support .* 100, err_lims, color = :red)
	f
end

# ╔═╡ 2c6551c0-65c9-4c79-866c-8c67ba191b43
md" ###### Used in later notebooks."

# ╔═╡ 0f567e25-2ce1-407d-8525-185e584de86a
begin
	death_raw=CSV.read(ros_datadir("Death", "dataforandy.csv"), DataFrame; missingstring="NA")
	death = death_raw[completecases(death_raw), :]
end

# ╔═╡ 0a7444c8-29ef-43e2-be65-3a6979d8315b
let
	st_abbr = death[:, 1]
	ex_rate = death[:, 8] ./ 100
	err_rate = death[:, 7] ./ 100
	hom_rate = death[:, 5] ./ 100000
	ds_per_homicide = death[:, 3] ./ 1000
	ds = death[:, 2]
	hom = ds ./ ds_per_homicide
	ex = ex_rate .* ds
	err = err_rate .* ds
	pop = hom ./ hom_rate
	std_err_rate = sqrt.( (err .+ 1) .* (ds .+ 1 .- err) ./ ((ds .+ 2).^2 .* (ds .+ 3)) )
end;

# ╔═╡ Cell order:
# ╟─a0fa3631-557c-4bb4-8862-cd21e24655e1
# ╟─5a68738d-8f8f-44b1-af3c-f3ceed14d82b
# ╠═f96ef9cd-c3e2-4796-af8c-e5d94198fd6b
# ╠═1271ba57-93ff-4ef7-bfff-15c39a034b2c
# ╟─5356fcee-dada-48dd-b422-bed4b5595470
# ╠═71cd8293-8c62-42b3-a33e-def5f7192160
# ╟─8dba464a-3eb7-43e5-8324-85b938be6d51
# ╟─326e0fc7-978b-42ae-b532-bdafb0a4948a
# ╠═30832e8b-0642-447b-b281-207d7c4d73f4
# ╠═bf9d7328-9b2f-4465-a7c6-e011ebd0c86e
# ╠═c30cc031-16bd-4691-92e1-c89ceb5cc1ca
# ╠═3c9c37de-7254-4fa8-94ba-ee74045a5a97
# ╠═08be202b-f618-4a7f-a48a-5618b19495f5
# ╠═9a2ec98b-6f45-4bc2-8ac0-5d6337cfb644
# ╟─bc877661-9aa0-425d-88e0-b82239c2552c
# ╠═6e6c65b3-f7ad-4a7f-91fd-f065e7bd7ffe
# ╠═197b0a6a-7c1c-4aad-b9ba-5743c02169fc
# ╠═696dcadc-7b10-4095-b165-2dcde9f89001
# ╠═103de138-886b-446d-8752-255f9db9d978
# ╠═7c96df85-6981-45dc-a912-311e1c4fbad1
# ╠═9727012b-1b76-4cde-84ef-bdd7bfa4b025
# ╠═ec3f566f-278c-4342-a365-db379b7d54aa
# ╟─c752672e-56de-4498-a85d-e3dffdc254d6
# ╠═db99f612-77b4-4639-892b-389a20c0ae83
# ╠═da420651-95a6-4fdc-ad2f-dbe2f550f672
# ╠═821e8c88-627f-41c3-8128-3d3b859575e0
# ╠═a5cd0b09-eff2-42c5-ab13-39e003099f32
# ╠═1e38322c-821f-45a5-aad9-d4c114ae37b7
# ╠═444766e8-86b9-496b-8b11-979b08fa842e
# ╠═ae86b798-1d42-4bf0-8bde-7d4c922a48fd
# ╠═1e40f68d-49ea-4700-95b6-88f69e527036
# ╠═285f5142-6cfb-4038-843a-d800ceefff50
# ╠═b7390db2-ccf3-4c27-be14-01eaedef7574
# ╠═1a132ed9-3ff5-42d2-bb63-642958abc5ce
# ╠═aa015e5a-88ca-4699-8f39-d0c54d8679e5
# ╟─2bd08619-ccd4-4ade-8d40-955c5fcd3fc2
# ╟─4c04253e-cb54-4f46-817e-af76053eef16
# ╠═aee6f37f-2a8a-451b-96e4-ec4eeb852b20
# ╠═93d4f920-7e1a-473b-badc-49cfdcc5f456
# ╠═f78d1b5e-d713-4d51-918d-005e7ca845af
# ╠═958459f8-02be-4a84-a3a2-f5d144d21103
# ╠═c0aeefbd-db6f-4359-80c1-9ff6ef5bb6f3
# ╠═d3ae1909-f6a3-437a-9d19-5b4a6e6baab3
# ╟─2c6551c0-65c9-4c79-866c-8c67ba191b43
# ╠═0f567e25-2ce1-407d-8525-185e584de86a
# ╠═0a7444c8-29ef-43e2-be65-3a6979d8315b
