### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 5084b8f0-65ac-4704-b1fc-2a9008132bd7
using Pkg, DrWatson

# ╔═╡ 550371ad-d411-4e66-9d63-7329322c6ea1
begin
	# Specific to this notebook
    using GLM

	# Specific to ROSStanPluto
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
	Logging.disable_logging(Logging.Warn);
end

# ╔═╡ eb7ea04a-da52-4e69-ac3e-87dc7f014652
md"## See chapter 1 in Regression and Other Stories."

# ╔═╡ cf39df58-3371-4535-88e4-f3f6c0404500
md" ###### Widen the cells."

# ╔═╡ 0616ece8-ccf8-4281-bfed-9c1192edf88e
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

# ╔═╡ 4755dab0-d228-41d3-934a-56f2863a5652
md"###### A typical set of Julia packages to include in notebooks."

# ╔═╡ 87902f0e-5919-45b3-89a6-b88a7dab9363
md" ### 1.1 The three challenges of statistics."

# ╔═╡ 56aa0b49-1e8a-4390-904d-6f7551f849ea
md"
!!! note

It is not common for me to copy from the book but this particular section deserves an exception!"

# ╔═╡ 47a6a5f3-0a54-46fe-a581-7414c0d9294a
md"

The three challenges of statistical inference are:
1. Generalizing from sample to population, a problem that is associated with survey sampling but actually arises in nearly every application of statistical inference;
2. Generalizing from treatment to control group, a problem that is associated with causal inference, which is implicitly or explicitly part of the interpretation of most regressions we have seen; and
3. Generalizing from observed measurements to the underlying constructs of interest, as most of the time our data do not record exactly what we would ideally like to study.
All three of these challenges can be framed as problems of prediction (for new people or new items that are not in the sample, future outcomes under different potentially assigned treatments, and underlying constructs of interest, if they could be measured exactly).
"

# ╔═╡ 0391fc17-09b7-47d7-b799-6dc6de13e82b
md"### 1.2 Why learn regression?"

# ╔═╡ d830f41c-0fb6-4bff-9fe0-0bd51f444779
hibbs = CSV.read(ros_datadir("ElectionsEconomy", "hibbs.csv"), DataFrame)

# ╔═╡ 35bee056-5cd8-48ee-b9c0-74a8b53229bd
hibbs_lm = lm(@formula(vote ~ growth), hibbs)

# ╔═╡ 3c4672aa-d17e-4681-9863-9ee026fefee6
residuals(hibbs_lm)

# ╔═╡ a9970ef7-1e0e-4976-b8c9-1db4dd3a222b
mad(residuals(hibbs_lm))

# ╔═╡ f48df50b-5450-4998-8dab-014c8b9d42a2
std(residuals(hibbs_lm))

# ╔═╡ be41c745-c87d-4f3a-ab4e-a8ae3b9ae091
coef(hibbs_lm)

# ╔═╡ 06ab4f30-68cc-4e35-9fa2-b8f8f25d3776
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

# ╔═╡ fa2fe95b-fe29-40c8-8dfc-27a35e720f3d
md" ### 1.3 Some examples of regression."

# ╔═╡ accfc0d8-968a-4b6c-bc1b-9da1aebe6cde
md" #### Electric company"

# ╔═╡ 305f0fb9-5e3a-45fd-8f57-edfdf65fb0e8
begin
	electric = CSV.read(ros_datadir("ElectricCompany", "electric.csv"), DataFrame)
	electric = electric[:, [:post_test, :pre_test, :grade, :treatment]]
	electric.grade = categorical(electric.grade)
	electric.treatment = categorical(electric.treatment)
	electric
end

# ╔═╡ 43e020d2-063c-43da-b7b3-bbc989002e9e
md"##### A quick look at the overall values of `pre_test` and `post_test`."

# ╔═╡ 3bc6b063-6f3d-4474-99b2-c9270513778a
describe(electric)

# ╔═╡ 82c83206-d5d7-4cc2-b4cd-d43e9c84c68a
all(completecases(electric)) == true

# ╔═╡ 420b8920-5f2a-4e3b-a32e-622252b84444
md" ##### Post-test density for each grade conditioned on treatment."

# ╔═╡ 54e3c7b6-c2b0-47d0-890a-5c55a19e42d9
let
	f = Figure()
	axis = (; width = 150, height = 150)
	el = data(electric) * mapping(:post_test, col=:grade, color=:treatment)
	plt = el * AlgebraOfGraphics.histogram(;bins=20) * mapping(row=:treatment)
	draw!(f[1, 1], plt; axis)
	f
end

# ╔═╡ 3c03311f-bdb8-4c06-a870-3e70a628f684
let
	f = Figure()
	axis = (; width = 150, height = 150)
	el = data(electric) * mapping(:post_test, col=:grade, color=:treatment)
	plt = el * AlgebraOfGraphics.density() * mapping(row=:treatment)
	draw!(f[1, 1], plt; axis)
	f
end

# ╔═╡ fb1e8fd3-7217-4955-83bd-551693f1507b
md"
!!! note

In above cell, as density() is exported by both GLMakie and AlgebraOfGraphics, it needs to be qualified."

# ╔═╡ 30b7e449-bcc7-4dbe-aef3-a50b85048f03
let
	f = Figure()
	el = data(electric) * mapping(:post_test, col=:grade)
	plt = el * AlgebraOfGraphics.density() * mapping(color=:treatment)
	draw!(f[1, 1], plt)
	f
end

# ╔═╡ 093c1e47-00be-407e-83a4-0ac96be3262c
let
	plt = data(electric) * visual(Violin) * mapping(:grade, :post_test, dodge=:treatment, color=:treatment)
	draw(plt)
end

# ╔═╡ 35307905-cee1-4f35-a149-cdaaf7fc1294
md" #### Peacekeeping"

# ╔═╡ f4b870c6-240d-4a46-98c8-1a0dbe7dfc6b
peace = CSV.read(ros_datadir("PeaceKeeping", "peacekeeping.csv"), missingstring="NA", DataFrame)

# ╔═╡ 00f43b7d-2594-4433-a18f-92d9899fb014
describe(peace)

# ╔═╡ 0eb862b2-d3be-4626-a4e6-3a6bb736c960
md"##### A quick look at this Dates stuff!"

# ╔═╡ baa075dd-18cc-4fac-93ca-5b2011e54c26
peace.cfdate[1]

# ╔═╡ bf4e1ded-1e5e-4e8e-a027-106cc6836ed2
DateTime(1992, 4, 25)

# ╔═╡ 76e1793f-ad85-4714-9dde-4347f47a60fc
Date(1992, 8, 10) - Date(1992, 4, 25)

# ╔═╡ da2e1e8e-8477-4a1a-8bbe-a8a08b5f32ed
Date(1970,1,1)

# ╔═╡ 46e101c6-7d21-4a8a-b96d-3c58b4cdb992
Date(1970,1,1) + Dates.Day(8150)

# ╔═╡ 492f405f-bded-4a0c-9e2a-26c4eca588ce
Date(1992, 4, 25) - Date(1970, 1, 1)

# ╔═╡ 491daea4-d345-4167-a3f1-06669df7106c
peace.faildate[1] - peace.cfdate[1]

# ╔═╡ acda3b77-ccac-45a9-be64-c3747682629b
begin
	pks_df = peace[peace.peacekeepers .== 1, [:cfdate, :faildate]]
	nopks_df = peace[peace.peacekeepers .== 0, [:cfdate, :faildate]]
end;

# ╔═╡ 84723e78-2b12-4652-87eb-34b6026d5ff9
mean(peace.censored)

# ╔═╡ 43e79699-c47d-405b-b1db-eaa51d4fc2c4
length(unique(peace.war))

# ╔═╡ 29ef3b78-adb5-4248-b8d0-d745b3da0e2e
mean(peace[peace.peacekeepers .== 1, :censored])

# ╔═╡ c0f85cc0-fed0-40a4-887d-80d3ef8ebba6
mean(peace[peace.peacekeepers .== 0, :censored])

# ╔═╡ 99adcb27-1150-47f7-9ea3-4bc47c3382ff
mean(peace[peace.peacekeepers .== 1 .&& peace.censored .== 0, :delay])

# ╔═╡ bf4d2451-b429-4f87-9c0f-e4706b70f85c
mean(peace[peace.peacekeepers .== 0 .&& peace.censored .== 0, :delay])

# ╔═╡ 733e1631-b367-4668-81ff-d7518a502f99
median(peace[peace.peacekeepers .== 1 .&& peace.censored .== 0, :delay])

# ╔═╡ 12b83909-e746-42e0-848a-6ec92636f718
median(peace[peace.peacekeepers .== 0 .&& peace.censored .== 0, :delay])

# ╔═╡ 1182e0db-b7da-4233-a24a-27fa16e5c49b
let
	f = Figure()
	pks = peace[peace.peacekeepers .== 1 .&& peace.censored .== 0, :]
	nopks = peace[peace.peacekeepers .== 0 .&& peace.censored .== 0,:]
	
	for i in 1:2
		title = i == 1 ? "Peacekeepers" : "No peacekeepers"

		ax = Axis(f[i, 1]; title, xlabel="Years until return to war",
	    ylabel = "Frequency", yminorticksvisible = true,
		yminorgridvisible = true, yminorticks = IntervalsBetween(8))

		xlims!(ax, [0, 8])
		hist!(i == 1 ? pks.delay : nopks.delay)
	end
	f
end

# ╔═╡ 533858b2-0b7e-4f4f-ac3d-a3d49856ef4b
md"
!!! note

Censored means conflict had not returned until end of observation period (2004)."

# ╔═╡ edd2dbf1-ae2b-4453-a5e3-94b4a51be521
begin
	# Filter out missing badness rows.
	pb = peace[peace.badness .!== missing, :];	
	
	# Delays until return to war for uncensored, peacekeeper cases
	pks_uc = pb[pb.peacekeepers .== 1 .&& pb.censored .== 0, :delay]
	# Delays until return to war for censored, peacekeeper cases
	pks_c = pb[pb.peacekeepers .== 1 .&& pb.censored .== 1, :delay]

	# No peacekeepr cases.
	nopks_uc = pb[pb.peacekeepers .== 0 .&& pb.censored .== 0, :delay]
	nopks_c = pb[pb.peacekeepers .== 0 .&& pb.censored .== 1, :delay]

	# Crude measure (:badness) used for assessing situation
	badness_pks_uc = pb[pb.peacekeepers .== 1 .&& pb.censored .== 0, 
		:badness]
	badness_pks_c = pb[pb.peacekeepers .== 1 .&& pb.censored .== 1, 
		:badness]
	badness_nopks_uc = pb[pb.peacekeepers .== 0 .&& pb.censored .== 0, 
		:badness]
	badness_nopks_c = pb[pb.peacekeepers .== 0 .&& pb.censored .== 1, 
		:badness]
end;

# ╔═╡ 2ec2b2b1-f1d2-4cd5-a23f-2b80abc5d4cd
begin
	f = Figure()
	ax = Axis(f[1, 1], title = "With UN peacekeepers",
		xlabel = "Pre-treatment measure of problems in country", 
		ylabel = "Delay [yrs] before return to conflict")
	sca1 = scatter!(badness_pks_uc, pks_uc)
	sca2 = scatter!(badness_pks_c, pks_c)
	xlims!(ax, [-13, -2.5])
	Legend(f[1, 2], [sca1, sca2], ["Uncensored", "Censored"])
	ax.xticks = ([-12, -4], ["no so bad", "really bad"])

	
	ax = Axis(f[2, 1], title = "Without UN peacekeepers",
		xlabel = "Pre-treatment measure of problems in country", 
		ylabel = "Delay [yrs] before return to conflict")
	sca1 = scatter!(badness_nopks_uc, nopks_uc)
	sca2 = scatter!(badness_nopks_c, nopks_c)
	xlims!(ax, [-13, -2.5])
	Legend(f[2, 2], [sca1, sca2], ["Uncensored", "Censored"])
	ax.xticks = ([-12, -4], ["no so bad", "really bad"])

	f
end

# ╔═╡ 63917762-446c-4230-ae22-d42f0752ff36
md" ### 1.4 Challenges in building, understanding, and interpreting regression."

# ╔═╡ 783df69c-5368-4a9e-aabf-a46895712289
md" #### Simple causal"

# ╔═╡ 8f61506c-bccf-4614-b2ae-ce6379f71da7
md"
!!! note

In models like below I usually prefer to create 2 separate Stan Language models, one for the continuous case and another for the binary case. But they can be combined in a single model as shown below. I'm using this example to show one way to handle vectors returned from Stan's cmdstan."

# ╔═╡ 4e0a7086-a4af-4d3b-ae98-ec47b458e451
@model function ppl1_2a(x, y)
    a ~ Normal(10, 10)
    b ~ Normal(10, 10)
    σ ~ Exponential(1)
    μ = a .+ b .* x
    for i in eachindex(x)
        y[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ 4d494a8b-ca6f-4591-9943-de4eecd0c5c1
@model function ppl1_2b(x_binary, y)
    a ~ Normal(10, 10)
    b ~ Normal(10, 10)
    σ ~ Exponential(1)
    μ = a .+ b .* x_binary
    for i in eachindex(x_binary)
        y[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ f3b11aff-58cb-4a2d-9785-1a853dc22797
md"
!!! note

Aki Vehtari did not include a seed number in his code.
"

# ╔═╡ 7982c864-0269-47a3-aa94-d261a55730c7
begin
	Random.seed!(123)
	n = 50
	x = rand(Uniform(1, 5), n)
	x_binary = [x[i] < 3 ? 0 : 1 for i in 1:n]
	y = [rand(Normal(10 + 3x[i], 3), 1)[1] for i in 1:n]
end

# ╔═╡ 4f6f740c-9414-4ef3-b595-f09ab51609b7
begin
	m1_2at = ppl1_2a(x, y)
	chns1_2at = sample(m1_2at, NUTS(), MCMCThreads(), 1000, 4)
end

# ╔═╡ 6a946718-561c-499f-8035-acf910d9f950
describe(chns1_2at)

# ╔═╡ 942a32ff-1edd-4e84-a1fd-ace09d2b09ec
begin
	post1_2at = DataFrame(chns1_2at)[:, 3:5]
	ms1_2at = model_summary(post1_2at, names(post1_2at))
end

# ╔═╡ 9133f7a6-92a5-4775-9837-5ffa1948df15
begin
	m1_2bt = ppl1_2b(x_binary, y)
	chns1_2bt = sample(m1_2bt, NUTS(), MCMCThreads(), 1000, 4)
end

# ╔═╡ fef21b83-7dac-4145-88a1-6d40d611527b
describe(chns1_2bt)

# ╔═╡ 84520430-2bba-4194-b8c6-de5a53e53f23
begin
	post1_2bt = DataFrame(chns1_2bt)[:, 3:5]
	ms1_2bt = model_summary(post1_2bt, names(post1_2bt))
end

# ╔═╡ e5f22961-b531-4519-bfb0-a8196d77ba6c
let
	x1 = 1.0:0.01:5.0
	f = Figure()
	medians = ms1_2at[:, "median"]
	ax = Axis(f[1, 2], title = "Regression on continuous treatment",
		xlabel = "Treatment level", ylabel = "Outcome")
	sca1 = scatter!(x, y)
	annotations!("Slope of fitted line = $(round(medians[2], digits=2))",
		position = (2.8, 10), textsize=15)
	lin1 = lines!(x1, medians[1] .+ medians[2] * x1)

	x2 = 0.0:0.01:1.0
	medians = ms1_2bt[:, "median"]
	ax = Axis(f[1, 1], title="Regression on binary treatment",
		xlabel = "Treatment", ylabel = "Outcome")
	sca1 = scatter!(x_binary, y)
	lin1 = lines!(x2, medians[1] .+ medians[2] * x2)
	annotations!("Slope of fitted line = $(round(medians[2], digits=2))", 
		position = (0.4, 10), textsize=15)
	f
end

# ╔═╡ 0eeb634d-a05f-46b4-baa0-613ed841aeaa
@model function ppl1_3a(x, y)
    a ~ Normal(10, 5)
	b ~ Normal(0, 5)
    σ ~ Exponential(1)
    μ = a .+ b .* x
    for i in eachindex(x)
        y[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ 2b781e56-7e97-491f-912d-0d7d50b5ab08
@model function ppl1_3b(x, y)
    a ~ Normal(10, 5)
	b_exp ~ Normal(5, 5)
    σ ~ Exponential(1)
    μ = a .+ b_exp .* exp.(x)
    for i in eachindex(x)
        y[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ 6906be3f-51ed-4471-b843-7c7487c851c4
begin
	Random.seed!(1533)
	n1 = 30
	x1 = rand(Uniform(1, 5), n1)
	y1 = [rand(Normal(5 + 30exp(-x1[i]), 2), 1)[1] for i in 1:n1]
end;

# ╔═╡ a587d71c-6c6b-4298-8e16-0ac9a25041dc
begin
	m1_3at = ppl1_3a(x1, y1)
	chns1_3at = sample(m1_3at, NUTS(), MCMCThreads(), 1000, 4)
	post1_3at = DataFrame(chns1_3at[[:a, :b, :σ]])
	plot_chains(post1_3at, [:a, :b, :σ])
end

# ╔═╡ 0a4bc48c-be88-4305-8688-4504c4f8bab6
begin
	m1_3bt = ppl1_3b(x1, y1)
	chns1_3bt = sample(m1_3bt, NUTS(), MCMCThreads(), 1000, 4)
	post1_3bt = DataFrame(chns1_3bt[[:a, :b_exp, :σ]])
	plot_chains(post1_3bt, [:a, :b_exp, :σ])
end

# ╔═╡ dea5ae47-f8af-4b0d-ad9e-a479efd2e5a3
ms1_3at = model_summary(post1_3at, [:a, :b, :σ])

# ╔═╡ 24828730-3cdf-4f93-97d8-0fa431dc0572
ms1_3bt = model_summary(post1_3bt, [:a, :b_exp, :σ])

# ╔═╡ 8adff5cd-937c-4ebc-94eb-955f75e93097
â₁, b̂, σ̂₁ = ms1_3at[:, :median];

# ╔═╡ 571df463-1f3c-43d0-8c9e-e9b927694979
â₂, b̂ₑₓₚ, σ̂₂ = ms1_3bt[:, :median];

# ╔═╡ e17c5d8a-da92-4db6-a3d9-63265b2526d9
let
	x1 = range(1.0, 5.9, length=n1)
	f = Figure()
	ax = Axis(f[1, 1], title = "Linear regression",
		xlabel = "Treatments", ylabel = "Outcomes")
	scatter!(x1, y1)
	lines!(x1, â₁ .+ b̂ .* x1)

	ax = Axis(f[2, 1], title = "Non-linear regression",
		xlabel = "Treatments", ylabel = "Outcomes")
	scatter!(x1, y1)
	lines!(x1, â₂ .+ b̂ₑₓₚ .* exp.(-x1))
	f
end

# ╔═╡ bbdb71e9-01f8-4925-8707-b13df2917706
â₂

# ╔═╡ 6117eb00-9b5a-4a1e-bb5a-cf572ece8ee0
b̂ₑₓₚ

# ╔═╡ 0dfa1d16-25de-4eee-8595-1bc1425492ab
begin
	Random.seed!(12573)
	n2 = 100
	z = repeat([0, 1]; outer=50)
	df1_8 = DataFrame()
	df1_8.xx = [(z[i] == 0 ? rand(Normal(0, 1.2), 1).^2 : rand(Normal(0, 0.8), 1).^2)[1] for i in 1:n2]
	df1_8.z = z
	df1_8.yy = [rand(Normal(20 .+ 5df1_8.xx[i] .+ 10df1_8.z[i], 3), 1)[1] for i in 1:n2]
	df1_8
end

# ╔═╡ f43d2372-c868-4696-8702-1191e97c49f0
lm1_8 = lm(@formula(yy ~ xx + z), df1_8)

# ╔═╡ b670e644-67d9-4185-8402-6eda661f0d81
lm1_8_0 = lm(@formula(yy ~ xx), df1_8[df1_8.z .== 0, :])

# ╔═╡ 8a39f298-9ca3-4f59-964e-68a4de98128b
lm1_8_1 = lm(@formula(yy ~ xx), df1_8[df1_8.z .== 1, :])

# ╔═╡ 5e646a78-0ef7-4c8f-a316-b7c606e5e133
let
	â₁, b̂₁ = coef(lm1_8_0)
	â₂, b̂₂ = coef(lm1_8_1)
	x = range(0, maximum(df1_8.xx), length=40)
	
	f = Figure()
	ax = Axis(f[1, 1]; title="Figure 1.8")
	scatter!(df1_8.xx[df1_8.z .== 0], df1_8.yy[df1_8.z .== 0])
	scatter!(df1_8.xx[df1_8.z .== 1], df1_8.yy[df1_8.z .== 1])
	lines!(x, â₁ .+ b̂₁ * x, label = "Control")
	lines!(x, â₂ .+ b̂₂ * x, label = "Treated")
	axislegend(; position=(:right, :bottom))
	current_figure()
end

# ╔═╡ 08d695b3-8f77-4079-9106-3d38d9762cc3
md" ### 1.5 Classical and Bayesian inference."

# ╔═╡ 74b247dc-c058-433b-9881-e1b85dacae84
md" ### 1.6 Computing least-squares and Bayesian regression."

# ╔═╡ effd481c-a47f-404a-a42f-207528b9b41b
md" ### 1.8 Exercises."

# ╔═╡ 08710628-ff52-4a95-a4f5-5dfce2fda165
md" #### Helicopters"

# ╔═╡ b2045d0f-afc1-4046-90c5-55f39cf11c84
helicopters = CSV.read(ros_datadir("Helicopters", "helicopters.csv"), DataFrame)

# ╔═╡ 72f0a072-9c65-4a91-98b5-8967f2f6a5f3
md" ##### Simulate 40 helicopters."

# ╔═╡ f7444121-7211-4999-ac6b-3a3c8738a4e3
begin
	helis = DataFrame(width_cm = rand(Normal(5, 2), 40), length_cm = rand(Normal(10, 4), 40))
	helis.time_sec = 0.5 .+ 0.04 .* helis.width_cm .+ 0.08 .* helis.length_cm .+ 0.1 .* rand(Normal(0, 1), 40)
	helis
end

# ╔═╡ f43b4ef0-9e1a-4b80-8a83-b8b3ff497a16
md" ##### Simulate 40 helicopters."

# ╔═╡ b609b38b-4e71-482e-b12f-666c906fba38
@model function ppl1_4(w, l, y)
    a ~ Normal(10, 5)
    b ~ Normal(0, 5)
	c ~ Normal(0, 5)
    σ ~ Exponential(1)
    μ = a .+ b .* w .+ c .* l
    for i in eachindex(y)
        y[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ 4bfd6fe4-51cd-473a-967f-889f44e29cf8
begin
	m1_4t = ppl1_4(helis.width_cm, helis.length_cm, helis.time_sec)
	chns1_4t = sample(m1_4t, NUTS(), MCMCThreads(), 1000, 4)
	post1_4t = DataFrame(chns1_4t[[:a, :b, :c, :σ]])
	plot_chains(post1_4t, [:a, :b, :c])
end

# ╔═╡ b85fd691-be86-4b82-ac26-7de4ca6021e5
	ms1_4t = model_summary(post1_4t, [:a, :b, :c, :σ])

# ╔═╡ d16f2383-5863-4035-b818-ecdafee85fb7
plot_chains(post1_4t, [:a, :b, :c])

# ╔═╡ e1b0e312-de63-4c8d-bc3c-d6d507921d51
let
	w = 1.0:0.01:8.0
	l = 6.0:0.01:15.0
	f = Figure()
	ax = Axis(f[1, 1], title = "Time on width or width",
		xlabel = "Width/Length", ylabel = "Time in the air")
	lines!(w, mean(post1_4t.a) .+ mean(post1_4t.b) .* w .+ mean(post1_4t.c))
	lines!(l, mean(post1_4t.a) .+ mean(post1_4t.c) .* l .+ mean(post1_4t.b))

	current_figure()
end

# ╔═╡ 790839f9-0b49-4da2-8dc1-00bab883e3af
let
	w_range = LinRange(1.0, 8.0, 100)
	w_times = mean.(link(post1_4t, (r, w) -> r.a + r.c + r.b * w, w_range))
	l_range = LinRange(6.0, 15.0, 100)
	l_times = mean.(link(post1_4t, (r, l) -> r.a + r.b + r.c * l, l_range))
	
	f = Figure()
	ax = Axis(f[1, 1], title = "Time in the air on width and length",
		xlabel = "Width/Length", ylabel = "Time in the air")
	
	lines!(w_range, w_times; label="Width")
	lines!(l_range, l_times; label="Length")

	f[1, 2] = Legend(f, ax, "Regression lines", framevisible = false)
	
	current_figure()
end

# ╔═╡ 44f06f91-a6b2-4c84-8be1-a86bc2040ba3
md"
!!! note

Note that the `link` function is defined in both RegeressionAndOtherStories (ROS) and Turing. In this case I added the `import` statement at the top of this notebook but I could also have qualified the call to ling ( `ROS.link` )."

# ╔═╡ b518fea5-298c-46f0-a749-4238ba2af17f
lnk1_4t = link(post1_4t, (r, l) -> r.a + r.b + r.c * l, [5, 10,12])

# ╔═╡ 7434c4d4-b398-41c3-ab52-1b1b8a4b4f72
median.(lnk1_4t)

# ╔═╡ 07ea7258-f35d-439a-9c20-71e4b95df808
mad.(lnk1_4t)

# ╔═╡ b413c2d6-dc44-4437-8123-ee7793863387
mean.(link(post1_4t, (r, l) -> r.a + r.b + r.c * l, [5, 10,12]))

# ╔═╡ Cell order:
# ╟─eb7ea04a-da52-4e69-ac3e-87dc7f014652
# ╟─cf39df58-3371-4535-88e4-f3f6c0404500
# ╠═0616ece8-ccf8-4281-bfed-9c1192edf88e
# ╟─4755dab0-d228-41d3-934a-56f2863a5652
# ╠═5084b8f0-65ac-4704-b1fc-2a9008132bd7
# ╠═550371ad-d411-4e66-9d63-7329322c6ea1
# ╟─87902f0e-5919-45b3-89a6-b88a7dab9363
# ╟─56aa0b49-1e8a-4390-904d-6f7551f849ea
# ╟─47a6a5f3-0a54-46fe-a581-7414c0d9294a
# ╟─0391fc17-09b7-47d7-b799-6dc6de13e82b
# ╠═d830f41c-0fb6-4bff-9fe0-0bd51f444779
# ╠═35bee056-5cd8-48ee-b9c0-74a8b53229bd
# ╠═3c4672aa-d17e-4681-9863-9ee026fefee6
# ╠═a9970ef7-1e0e-4976-b8c9-1db4dd3a222b
# ╠═f48df50b-5450-4998-8dab-014c8b9d42a2
# ╠═be41c745-c87d-4f3a-ab4e-a8ae3b9ae091
# ╠═06ab4f30-68cc-4e35-9fa2-b8f8f25d3776
# ╟─fa2fe95b-fe29-40c8-8dfc-27a35e720f3d
# ╟─accfc0d8-968a-4b6c-bc1b-9da1aebe6cde
# ╠═305f0fb9-5e3a-45fd-8f57-edfdf65fb0e8
# ╟─43e020d2-063c-43da-b7b3-bbc989002e9e
# ╠═3bc6b063-6f3d-4474-99b2-c9270513778a
# ╠═82c83206-d5d7-4cc2-b4cd-d43e9c84c68a
# ╟─420b8920-5f2a-4e3b-a32e-622252b84444
# ╠═54e3c7b6-c2b0-47d0-890a-5c55a19e42d9
# ╠═3c03311f-bdb8-4c06-a870-3e70a628f684
# ╟─fb1e8fd3-7217-4955-83bd-551693f1507b
# ╠═30b7e449-bcc7-4dbe-aef3-a50b85048f03
# ╠═093c1e47-00be-407e-83a4-0ac96be3262c
# ╟─35307905-cee1-4f35-a149-cdaaf7fc1294
# ╠═f4b870c6-240d-4a46-98c8-1a0dbe7dfc6b
# ╠═00f43b7d-2594-4433-a18f-92d9899fb014
# ╟─0eb862b2-d3be-4626-a4e6-3a6bb736c960
# ╠═baa075dd-18cc-4fac-93ca-5b2011e54c26
# ╠═bf4e1ded-1e5e-4e8e-a027-106cc6836ed2
# ╠═76e1793f-ad85-4714-9dde-4347f47a60fc
# ╠═da2e1e8e-8477-4a1a-8bbe-a8a08b5f32ed
# ╠═46e101c6-7d21-4a8a-b96d-3c58b4cdb992
# ╠═492f405f-bded-4a0c-9e2a-26c4eca588ce
# ╠═491daea4-d345-4167-a3f1-06669df7106c
# ╠═acda3b77-ccac-45a9-be64-c3747682629b
# ╠═84723e78-2b12-4652-87eb-34b6026d5ff9
# ╠═43e79699-c47d-405b-b1db-eaa51d4fc2c4
# ╠═29ef3b78-adb5-4248-b8d0-d745b3da0e2e
# ╠═c0f85cc0-fed0-40a4-887d-80d3ef8ebba6
# ╠═99adcb27-1150-47f7-9ea3-4bc47c3382ff
# ╠═bf4d2451-b429-4f87-9c0f-e4706b70f85c
# ╠═733e1631-b367-4668-81ff-d7518a502f99
# ╠═12b83909-e746-42e0-848a-6ec92636f718
# ╠═1182e0db-b7da-4233-a24a-27fa16e5c49b
# ╟─533858b2-0b7e-4f4f-ac3d-a3d49856ef4b
# ╠═edd2dbf1-ae2b-4453-a5e3-94b4a51be521
# ╠═2ec2b2b1-f1d2-4cd5-a23f-2b80abc5d4cd
# ╟─63917762-446c-4230-ae22-d42f0752ff36
# ╟─783df69c-5368-4a9e-aabf-a46895712289
# ╟─8f61506c-bccf-4614-b2ae-ce6379f71da7
# ╠═4e0a7086-a4af-4d3b-ae98-ec47b458e451
# ╠═4d494a8b-ca6f-4591-9943-de4eecd0c5c1
# ╟─f3b11aff-58cb-4a2d-9785-1a853dc22797
# ╠═7982c864-0269-47a3-aa94-d261a55730c7
# ╠═4f6f740c-9414-4ef3-b595-f09ab51609b7
# ╠═6a946718-561c-499f-8035-acf910d9f950
# ╠═942a32ff-1edd-4e84-a1fd-ace09d2b09ec
# ╠═9133f7a6-92a5-4775-9837-5ffa1948df15
# ╠═fef21b83-7dac-4145-88a1-6d40d611527b
# ╠═84520430-2bba-4194-b8c6-de5a53e53f23
# ╠═e5f22961-b531-4519-bfb0-a8196d77ba6c
# ╠═0eeb634d-a05f-46b4-baa0-613ed841aeaa
# ╠═2b781e56-7e97-491f-912d-0d7d50b5ab08
# ╠═6906be3f-51ed-4471-b843-7c7487c851c4
# ╠═a587d71c-6c6b-4298-8e16-0ac9a25041dc
# ╠═0a4bc48c-be88-4305-8688-4504c4f8bab6
# ╠═dea5ae47-f8af-4b0d-ad9e-a479efd2e5a3
# ╠═24828730-3cdf-4f93-97d8-0fa431dc0572
# ╠═8adff5cd-937c-4ebc-94eb-955f75e93097
# ╠═571df463-1f3c-43d0-8c9e-e9b927694979
# ╠═e17c5d8a-da92-4db6-a3d9-63265b2526d9
# ╠═bbdb71e9-01f8-4925-8707-b13df2917706
# ╠═6117eb00-9b5a-4a1e-bb5a-cf572ece8ee0
# ╠═0dfa1d16-25de-4eee-8595-1bc1425492ab
# ╠═f43d2372-c868-4696-8702-1191e97c49f0
# ╠═b670e644-67d9-4185-8402-6eda661f0d81
# ╠═8a39f298-9ca3-4f59-964e-68a4de98128b
# ╠═5e646a78-0ef7-4c8f-a316-b7c606e5e133
# ╟─08d695b3-8f77-4079-9106-3d38d9762cc3
# ╟─74b247dc-c058-433b-9881-e1b85dacae84
# ╟─effd481c-a47f-404a-a42f-207528b9b41b
# ╟─08710628-ff52-4a95-a4f5-5dfce2fda165
# ╠═b2045d0f-afc1-4046-90c5-55f39cf11c84
# ╟─72f0a072-9c65-4a91-98b5-8967f2f6a5f3
# ╠═f7444121-7211-4999-ac6b-3a3c8738a4e3
# ╟─f43b4ef0-9e1a-4b80-8a83-b8b3ff497a16
# ╠═b609b38b-4e71-482e-b12f-666c906fba38
# ╠═4bfd6fe4-51cd-473a-967f-889f44e29cf8
# ╠═b85fd691-be86-4b82-ac26-7de4ca6021e5
# ╠═d16f2383-5863-4035-b818-ecdafee85fb7
# ╠═e1b0e312-de63-4c8d-bc3c-d6d507921d51
# ╠═790839f9-0b49-4da2-8dc1-00bab883e3af
# ╟─44f06f91-a6b2-4c84-8be1-a86bc2040ba3
# ╠═b518fea5-298c-46f0-a749-4238ba2af17f
# ╠═7434c4d4-b398-41c3-ab52-1b1b8a4b4f72
# ╠═07ea7258-f35d-439a-9c20-71e4b95df808
# ╠═b413c2d6-dc44-4437-8123-ee7793863387
