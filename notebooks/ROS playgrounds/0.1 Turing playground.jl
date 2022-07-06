### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ bf1dcdac-9a73-41c1-be63-334b86a4fd93
using Pkg

# ╔═╡ 1dc0d3f1-a1fc-407a-a857-1e77a61aa95e
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
	#Logging.disable_logging(Logging.Warn)
end;

# ╔═╡ 1f693212-3ac6-4ddf-a524-b571ab062a9e
md"#### In `Regression and Other Stories`, mcmc is _just_ a tool. Hence whether one uses Stan or Turing is not the main focus of the book. This notebook uses `ElectionsEconomy: hibbs.csv` to illustrate how Turing and other computations are used in the Julia _project_ ROSTuringPluto.jl."

# ╔═╡ 32a2652f-045b-4c06-8b68-b9420b1eac42
md"##### See also Chapter 1.2, Figure 1.1 in Regression and Other Stories."

# ╔═╡ f693f1fa-1c8f-4274-b9fe-0d3eb8865735
md" ##### Over time I will expand below the list of topics:

1. Turing
2. Using median and mad to summarize a posterior distribution.
3. ...
4. Model comparisons (TBD)
5. DAGs (TBD)
6. Graphs (TBD)
7. ...

"

# ╔═╡ edc8ea2f-53c6-4c91-9f3d-9278fac1b89c
md" ##### Widen the cells."

# ╔═╡ 64f79bec-b35c-465f-ab94-fe42170bb081
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

# ╔═╡ e4d188b3-e77e-4872-9341-9a01aea4d9bd
md"###### A typical set of Julia packages to include in notebooks."

# ╔═╡ 24217b93-c088-4ce5-890b-138f6b8a16da
md"
!!! note

All data files are available (as .csv files) in the data subdirectory of package RegressionAndOtherStories.jl.
"

# ╔═╡ 062e11bb-b5e1-4093-a811-195a7f8eedf1
ros_datadir()

# ╔═╡ 4ea0c8a8-32e8-4a4e-ae02-e942def56816
md"
!!! note

After evaluating above cell, use `ros_datadir(\"ElectionsEconomy\", \"hibbs.dat\")` to obtain data."

# ╔═╡ 968f8ef9-afaa-459d-bdbc-ce71a8ca0c93
hibbs = CSV.read(ros_datadir("ElectionsEconomy", "hibbs.csv"), DataFrame)

# ╔═╡ 10e77828-8f8c-48a2-81a2-ee5a64e33d34
hibbs_lm = lm(@formula(vote ~ growth), hibbs)

# ╔═╡ 3ec41299-4db8-4674-930d-020c364712f5
residuals(hibbs_lm)

# ╔═╡ 95c8c2cc-8de8-4477-9178-96df20b46c34
mad(residuals(hibbs_lm))

# ╔═╡ f4ac50f6-9024-43be-9014-1dbee2818e6e
std(residuals(hibbs_lm))

# ╔═╡ 433b3627-2796-4720-b59d-2b694f20c096
coef(hibbs_lm)

# ╔═╡ 54b5b58f-91d5-4354-92d0-608fbd5adb64
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

# ╔═╡ f4a8b816-f700-4438-8fbb-a6fdb55e0cf0
md" #### Below some additional cells demonstrating the use of Turing."

# ╔═╡ d5d66e16-487e-4214-b702-728dc3db32cc
@model function ppl1_1(growth, vote)
    a ~ Normal(50, 20)
    b ~ Normal(0, 5)
    σ ~ Exponential(1)
    μ = a .+ b .* growth
    for i in eachindex(vote)
        vote[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ b14217e8-1f70-44cc-8d3c-528ab55e27ef
md"
!!! note

The sequence of the statements matter in Turing models!"

# ╔═╡ b3adef33-c111-4d20-b277-a5346eae9f23
begin
	m1_1t = ppl1_1(hibbs.growth, hibbs.vote)
	chns1_1t = sample(m1_1t, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns1_1t)
end

# ╔═╡ 13e24eac-d6b1-490d-840c-eec58ed08503
md"
!!! note

Mostly I disable logging early on in notebooks using Turing. But it is also possible to do this `by cell`. Click on the little circle with 3 dots at the top of the selected cell and select `Hide logs`."

# ╔═╡ ff355e32-acfd-44bc-ba5c-85d925c98aff
begin
	post1_1t = DataFrame(chns1_1t)[:, [:a, :b, :σ]]
end

# ╔═╡ f52534b9-80e8-4d55-a826-8deccd1c0ee6
describe(chns1_1t)

# ╔═╡ 26d0ea8c-c405-4013-9159-d84690a85080
plot_chains(post1_1t, [:a, :b, :σ])

# ╔═╡ 1f4e22a4-5fba-4331-9bf9-dd886ce45e74
let
	x = -1.0:0.1:6.0
	preds = mean(post1_1t.a) .+ mean(post1_1t.b) .* x
	lines(x, preds, color=:darkblue, label="Regression line")
	scatter!(hibbs.growth, hibbs.vote, marker=:cross, markersize=10,
		color=:darkred, label="Observations")
	current_figure()
end

# ╔═╡ 75ee3adf-2988-4f42-a0b5-21abe5829120
md" ### Priors of Turing models."

# ╔═╡ 63db6c9f-ba40-41a2-975d-798d9f2cbeab
begin
	prior_chns1_1t = sample(m1_1t, Prior(), 1000)
	describe(prior_chns1_1t)
end

# ╔═╡ 3f5d4c7d-73ea-41c5-bc9c-ff91b6794444
let
	N = 100
	x = LinRange(-1, 4, N)
	priors4_1t = DataFrame(prior_chns1_1t)

	mat1 = zeros(50, 100)
	for i in 1:50
		mat1[i, :] = priors4_1t.a[i] .+ priors4_1t.b[i] .* x
	end
	ā = mean(post1_1t.a)
	b̄ = mean(post1_1t.b)

	# Maybe could use a `link` function here
	mat2 = zeros(50, 100)
	for i in 1:50
		mat2[i, :] = post1_1t.a[i] .+ post1_1t.b[i] .* x
	end

	fig = Figure()
	xlabel = "Average growth personal income [%]"
	ylabel="Incumbent's party vote share"
	ax = Axis(fig[1, 1]; title="Lines based on prior samples", 
		xlabel, ylabel)
	ylims!(ax, 40, 65)
	series!(fig[1, 1], x, mat1, solid_color=:lightgrey)
	ax = Axis(fig[1, 2]; title="Lines based on posterior samples", 
		xlabel, ylabel)
	ylims!(ax, 40, 65)
	series!(fig[1, 2], x, mat2, solid_color=:lightgrey)
	scatter!(hibbs.growth, hibbs.vote)
	lines!(fig[1, 2], x, ā .+ b̄ * x, color = :red)

	fig
end

# ╔═╡ 534edcc4-2dac-4ddb-a6a9-55ee38a3e7ae
md" ### Quadratic approximation."

# ╔═╡ 7c8e8a61-b1c1-4f9a-addc-61a987c2c5d2
map_estimate = optimize(m1_1t, MAP())

# ╔═╡ a9d5118d-8fc7-4353-bb20-6b87ae0e9247
begin
	â, b̂, σ̂ = map_estimate.values
	(â = â, b̂ = b̂, σ̂ = σ̂)
end

# ╔═╡ f41d6486-7589-4b67-9eaf-92956622226a
let
	x = -1:0.1:6
	preds = mean(post1_1t.a) .+ mean(post1_1t.b) .* x
	f = Figure()
	ax = Axis(f[1, 1], title = "Regression line (green) and MAP based regression line (blue).",
		xlabel = "Growth", ylabel = "Vote")
	
	lines!(f[1, 1], x, â .+ b̂ .* x, color=:darkblue)
	lines!(x, preds, color=:darkgreen, label="Regression line")
	scatter!(hibbs.growth, hibbs.vote, color=:darkred, leg=false)
	current_figure()
end

# ╔═╡ 164949ac-1246-4693-adba-ae46df4c9f1c
md" ### Prediction"

# ╔═╡ c639817a-ac0c-4238-b088-6dde80cf8b80
begin
	x_test = [0, 1, 2, 3, 4, 5]
	m_test = ppl1_1(x_test, fill(missing, length(x_test)))
	pred_chns1_1t = predict(m_test, chns1_1t)
	pred_chns1_1t
end


# ╔═╡ 00781862-dc39-4623-84c9-ce214632ae08
describe(pred_chns1_1t)

# ╔═╡ 7195be8a-69dc-4efe-8737-30d2c52c5f2e
begin
	pred1_1t = DataFrame(pred_chns1_1t)[:, 3:end]
	ms1_1t = model_summary(pred1_1t, names(pred1_1t))
end

# ╔═╡ dab4aa24-1240-4799-a6e1-7474c6a8d2bc
pred1_1t

# ╔═╡ 364e4670-25fb-4dd2-97fe-40746ecd0029
let
	x = -1:0.1:6
	preds = mean(post1_1t.a) .+ mean(post1_1t.b) .* x
	f = Figure()
	ax = Axis(f[1, 1], title = "Regression line (green) and MAP based regression line (blue).",
		xlabel = "Growth", ylabel = "Vote")
	
	lines!(f[1, 1], x, â .+ b̂ .* x, color=:darkblue)
	lines!(x, preds, color=:darkgreen, label="Regression line")
	scatter!(hibbs.growth, hibbs.vote, color=:darkred, leg=false)
	scatter!(x_test, reshape(mean(Matrix(pred1_1t); dims=1), ncol(pred1_1t)), markersize=20)
	current_figure()
end

# ╔═╡ 4faa2a07-292a-498e-a2d7-74848e9826de
function nested_column_to_matrix(df::DataFrame, var::Union{Symbol, String})
	m = zeros(nrow(df), length(df[1, var]))
	i = 1
    for r in eachrow(df[:, var])
		for j in 1:length(r[1])
			m[i, j] = r[1][j]
		end
		i += 1
    end
	m
 end

# ╔═╡ 5ba9b7d1-7926-435e-95e7-beefbd440c35
function errorbars_mean(df, p = [0.055, 0.945])
	se_df = DataFrame()
	i = 1
	for col in eachcol(df)
		n = length(col)
		est = mean(col)
		se = std(col)/sqrt(n)
		int = [abs.(quantile.(TDist(n-1), p) * se)]
		append!(se_df, DataFrame(parameters = names(df)[i], estimate = est, se = se, p = [p], q = int))
		i += 1
	end
	se_df
end

# ╔═╡ 7cb60d65-0dc1-432f-b023-1ce592a5a174
errorbars_mean(pred1_1t)

# ╔═╡ f978a052-90e1-4166-aea4-8f23ccb681d3
function errorbars_draws(df, p = [0.25, 0.75])
	q_df = DataFrame()
	i = 1
	for col in eachcol(df)
    	m = median(col)
		s = mad(col)
    	int = [abs.(quantile(col, p) .- m)]
		append!(q_df, DataFrame(parameters = names(df)[i], median = m, mad_sd = s, p = [p], q = int))
		i += 1
	end
	q_df
end


# ╔═╡ 45fb3b70-7e67-42fb-aa67-067e06ab68aa
errorbars_draws(pred1_1t, [0.055, 0.945])

# ╔═╡ 4d1ecd0a-337d-4507-8e38-8195f80d6d99
let
	x = -1:0.1:6
	preds = mean(post1_1t.a) .+ mean(post1_1t.b) .* x
	pred_values = reshape(mean(Matrix(pred1_1t); dims=1), ncol(pred1_1t))

	f = Figure()
	ax = Axis(f[1, 1], title = "Regression line (green).",
		xlabel = "Growth", ylabel = "Vote")
	
	lines!(x, preds, color=:darkgreen, label="Regression line")
	scatter!(hibbs.growth, hibbs.vote, color=:darkred, leg=false)

	# 50% interval predictions
	error_bars = nested_column_to_matrix(errorbars_draws(pred1_1t, [0.25, 0.75]), :q)
	errorbars!(x_test, pred_values, error_bars[:, 1], error_bars[:, 2], whiskerwidth = 6, color=:grey)

	# 89% s.e. of the mean
	error_bars = nested_column_to_matrix(errorbars_mean(pred1_1t, [0.055, 0.945]), :q)
	errorbars!(x_test, pred_values, error_bars[:, 1], error_bars[:, 2], whiskerwidth = 6, color=:black)
	current_figure()
end

# ╔═╡ 10eb547d-f763-4dfd-8e90-aec59b398823
md" ###### A quick look at broadcasting and vectorization. See also [more dots](https://julialang.org/blog/2017/01/moredots/) "

# ╔═╡ f42f37ec-ba00-4053-827e-7a96eb4cc4a4
f(x) = 3x^2 + 5x + 2

# ╔═╡ cb3f7b9d-4438-42a6-b62c-f6ebdf86c134
function nobcst(f, x)
	f.(2 .* x.^2 .+ 6 .* x.^3 .- sqrt.(x))
end

# ╔═╡ 7f96fd3e-e37b-4bd9-9d8f-40a8e816580a
function bcst(f, x)
	@. f(2 * x^2 + 6 * x^3 - sqrt(x))
end

# ╔═╡ 98d9d553-8575-4a92-96b6-4e4ecbc99ab4
let
	n = 10^6
	x = LinRange(0, 2, n)
	@time nobcst(f, x)
end

# ╔═╡ daeae938-a081-4e36-ade7-1e610ac94217
let
	n = 10^6
	x = LinRange(0, 2, n)
	@time bcst(f, x)
end

# ╔═╡ 7479e897-1eb3-4251-9d3e-f97e7cf771fd
md"#### Compute median and mad."

# ╔═╡ 130040e5-620f-464b-a8b9-ced5fa6caa6f
ms1_1t["vote[2]", "mad_sd"]

# ╔═╡ 97df9584-1e89-43e4-9b14-db4eb3fd3c1b
md" ##### Alternative computation of mad()."

# ╔═╡ 8fe0efd4-1890-4912-baed-231010ee8744
let
	1.483 .* [median(abs.(post1_1t.a .- median(post1_1t.a))),
	median(abs.(post1_1t.b .- median(post1_1t.b))),
	median(abs.(post1_1t.σ .- median(post1_1t.σ)))]
end

# ╔═╡ 42ee8c7a-05b7-4a97-bb2c-9eb60be8102e
md" ##### Quick simulation with median, mad, mean and std of Normal observations."

# ╔═╡ 983cde20-cfc3-4241-a227-ef74f29fa075
nt = (x=rand(Normal(5, 2), 10000),)

# ╔═╡ 00baf816-be7e-4a44-9662-63c11282cad6
[median(nt.x), mad(nt.x), mean(nt.x), std(nt.x)]

# ╔═╡ 8c962cfd-4584-4c88-85bc-6a1c2b7d2150
sd_mean = round(mad(nt.x)/√10000; digits=2)

# ╔═╡ 59c24421-58d1-4964-abc1-d0192c1d3baa
median(abs.(nt.x .- median(nt.x)))

# ╔═╡ 7159c229-5d6a-4773-8da6-ab327fbbd7df
1.483 * median(abs.(nt.x .- median(nt.x)))

# ╔═╡ 0e4afc23-5676-401f-a60d-2eab4320abb0
let
	plt = data(nt) * mapping(:x) * AlgebraOfGraphics.density()
	axis = (; title="Density x")
	draw(plt; axis)
end

# ╔═╡ d7ddd741-fff5-4bdf-8d54-58ec6fd5d90d
quantile(nt.x, [0.025, 0.975])

# ╔═╡ 5c7b1814-a934-491a-904e-35c0b6db2eb4
quantile(nt.x, [0.25, 0.75])

# ╔═╡ ccf333ff-af94-439a-b41c-5be04c076007
md"""
!!! note

Click on "Live docs" and place cursor on link to see more help. 

Click little down arrow to the right to remove live docs again.
"""

# ╔═╡ Cell order:
# ╟─1f693212-3ac6-4ddf-a524-b571ab062a9e
# ╟─32a2652f-045b-4c06-8b68-b9420b1eac42
# ╟─f693f1fa-1c8f-4274-b9fe-0d3eb8865735
# ╟─edc8ea2f-53c6-4c91-9f3d-9278fac1b89c
# ╠═64f79bec-b35c-465f-ab94-fe42170bb081
# ╟─e4d188b3-e77e-4872-9341-9a01aea4d9bd
# ╠═bf1dcdac-9a73-41c1-be63-334b86a4fd93
# ╠═1dc0d3f1-a1fc-407a-a857-1e77a61aa95e
# ╟─24217b93-c088-4ce5-890b-138f6b8a16da
# ╠═062e11bb-b5e1-4093-a811-195a7f8eedf1
# ╟─4ea0c8a8-32e8-4a4e-ae02-e942def56816
# ╠═968f8ef9-afaa-459d-bdbc-ce71a8ca0c93
# ╠═10e77828-8f8c-48a2-81a2-ee5a64e33d34
# ╠═3ec41299-4db8-4674-930d-020c364712f5
# ╠═95c8c2cc-8de8-4477-9178-96df20b46c34
# ╠═f4ac50f6-9024-43be-9014-1dbee2818e6e
# ╠═433b3627-2796-4720-b59d-2b694f20c096
# ╠═54b5b58f-91d5-4354-92d0-608fbd5adb64
# ╟─f4a8b816-f700-4438-8fbb-a6fdb55e0cf0
# ╠═d5d66e16-487e-4214-b702-728dc3db32cc
# ╟─b14217e8-1f70-44cc-8d3c-528ab55e27ef
# ╠═b3adef33-c111-4d20-b277-a5346eae9f23
# ╟─13e24eac-d6b1-490d-840c-eec58ed08503
# ╠═ff355e32-acfd-44bc-ba5c-85d925c98aff
# ╠═f52534b9-80e8-4d55-a826-8deccd1c0ee6
# ╠═26d0ea8c-c405-4013-9159-d84690a85080
# ╠═1f4e22a4-5fba-4331-9bf9-dd886ce45e74
# ╟─75ee3adf-2988-4f42-a0b5-21abe5829120
# ╠═63db6c9f-ba40-41a2-975d-798d9f2cbeab
# ╠═3f5d4c7d-73ea-41c5-bc9c-ff91b6794444
# ╟─534edcc4-2dac-4ddb-a6a9-55ee38a3e7ae
# ╠═7c8e8a61-b1c1-4f9a-addc-61a987c2c5d2
# ╠═a9d5118d-8fc7-4353-bb20-6b87ae0e9247
# ╠═f41d6486-7589-4b67-9eaf-92956622226a
# ╟─164949ac-1246-4693-adba-ae46df4c9f1c
# ╠═c639817a-ac0c-4238-b088-6dde80cf8b80
# ╠═00781862-dc39-4623-84c9-ce214632ae08
# ╠═7195be8a-69dc-4efe-8737-30d2c52c5f2e
# ╠═dab4aa24-1240-4799-a6e1-7474c6a8d2bc
# ╠═364e4670-25fb-4dd2-97fe-40746ecd0029
# ╠═4faa2a07-292a-498e-a2d7-74848e9826de
# ╠═5ba9b7d1-7926-435e-95e7-beefbd440c35
# ╠═7cb60d65-0dc1-432f-b023-1ce592a5a174
# ╠═f978a052-90e1-4166-aea4-8f23ccb681d3
# ╠═45fb3b70-7e67-42fb-aa67-067e06ab68aa
# ╠═4d1ecd0a-337d-4507-8e38-8195f80d6d99
# ╟─10eb547d-f763-4dfd-8e90-aec59b398823
# ╠═f42f37ec-ba00-4053-827e-7a96eb4cc4a4
# ╠═cb3f7b9d-4438-42a6-b62c-f6ebdf86c134
# ╠═7f96fd3e-e37b-4bd9-9d8f-40a8e816580a
# ╠═98d9d553-8575-4a92-96b6-4e4ecbc99ab4
# ╠═daeae938-a081-4e36-ade7-1e610ac94217
# ╟─7479e897-1eb3-4251-9d3e-f97e7cf771fd
# ╠═130040e5-620f-464b-a8b9-ced5fa6caa6f
# ╟─97df9584-1e89-43e4-9b14-db4eb3fd3c1b
# ╠═8fe0efd4-1890-4912-baed-231010ee8744
# ╟─42ee8c7a-05b7-4a97-bb2c-9eb60be8102e
# ╠═983cde20-cfc3-4241-a227-ef74f29fa075
# ╠═00baf816-be7e-4a44-9662-63c11282cad6
# ╠═8c962cfd-4584-4c88-85bc-6a1c2b7d2150
# ╠═59c24421-58d1-4964-abc1-d0192c1d3baa
# ╠═7159c229-5d6a-4773-8da6-ab327fbbd7df
# ╠═0e4afc23-5676-401f-a60d-2eab4320abb0
# ╠═d7ddd741-fff5-4bdf-8d54-58ec6fd5d90d
# ╠═5c7b1814-a934-491a-904e-35c0b6db2eb4
# ╟─ccf333ff-af94-439a-b41c-5be04c076007
