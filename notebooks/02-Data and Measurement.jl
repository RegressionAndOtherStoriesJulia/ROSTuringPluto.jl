### A Pluto.jl notebook ###
# v0.19.10

using Markdown
using InteractiveUtils

# ╔═╡ 5084b8f0-65ac-4704-b1fc-2a9008132bd7
using Pkg, DrWatson

# ╔═╡ f71640c9-3918-475e-b32b-c85424bbcf5e
begin
	# Specific to this notebook
    using GLM

	# Specific to ROSTuringPluto
	using Optim
	using Logging
    using Turing
	
	# Graphics related
    using GLMakie

	# Include basic packages
	using RegressionAndOtherStories
	import RegressionAndOtherStories: link

	Logging.disable_logging(Logging.Warn);
end;

# ╔═╡ eb7ea04a-da52-4e69-ac3e-87dc7f014652
md"## See chapter 2 in Regression and Other Stories."

# ╔═╡ d7543b63-52d3-449b-8ce3-d979c23f8b95
md" ###### Widen the notebook."

# ╔═╡ ed172871-fa4d-4111-ac0a-341898917948
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

# ╔═╡ ab84df44-ddde-4b7e-9370-b22646fe0a3b
md" ### 2.1 Examining where data come from."

# ╔═╡ 6c2043f4-dcbd-4535-a5ae-61c109519879
hdi = CSV.read(ros_datadir("HDI", "hdi.csv"), DataFrame)

# ╔═╡ d7b5bfec-3ca7-46fe-aed5-094ee266016b
let
	f = Figure()
	ax = Axis(f[1, 1]; title = "HDI",
		xlabel = "HDI rank of state", ylabel = "HDI index")
	limits!(ax, 0, 60, 0.7, 1)
	scatter!(hdi.rank, hdi.hdi)
	selection = 1:20:50
	scatter!(hdi.rank[selection], hdi.hdi[selection]; color=:darkred)
	for i in selection
		lines!([hdi.rank[i], hdi.rank[i] + 3], 
			[hdi.hdi[i], hdi.hdi[i] + 0.015]; color=:grey)
		annotations!(hdi.state[i], 
			position = (hdi.rank[i] + 3, hdi.hdi[i] + 0.015),
			textsize = 10)
	end
	selection = [4, 51]
	scatter!(hdi.rank[selection], hdi.hdi[selection]; color=:darkred)
	for i in selection
		lines!([hdi.rank[i], hdi.rank[i] + 3], 
			[hdi.hdi[i], hdi.hdi[i] - 0.015]; color=:grey)
		annotations!(hdi.state[i], 
			position = (hdi.rank[i] + 3, hdi.hdi[i] - 0.023),
			textsize = 10)
	end
	selection = 45:3:50
	scatter!(hdi.rank[selection], hdi.hdi[selection]; color=:darkred)
	for i in selection
		lines!([hdi.rank[i], hdi.rank[i] + 3], 
			[hdi.hdi[i], hdi.hdi[i] + 0.015]; color=:grey)
		annotations!(hdi.state[i], 
			position = (hdi.rank[i] + 3, hdi.hdi[i] + 0.015),
			textsize = 10)
	end
	f
end

# ╔═╡ 2dd08df7-a6dd-4a19-a673-a98c43e4d552
begin
	votes = CSV.read(ros_datadir("HDI", "votes.csv"), DataFrame; 
		delim=",", stringtype=String, pool=false)
	votes[votes.st_year .== 2000, [:st_state, :st_stateabb, :st_income]]
end

# ╔═╡ c4d6df3e-3cc9-4588-84e6-fe7265f01027
let
	tmp = votes[votes.st_year .== 2000, [:st_state, :st_stateabb, :st_income]]
	votes2 = DataFrame(state=tmp.st_state, abbr=tmp.st_stateabb,
		income=tmp.st_income)
	global hdivotes = innerjoin(hdi, votes2, on = :state)
end

# ╔═╡ 696eb9b0-f239-425b-8b71-802dfd9d8a42
let
	f = Figure()
	ax = Axis(f[1, 1]; title = "HDI ~ income",
		xlabel = "Average state income in 2000",
		ylabel = "Human Development Index")
	for i in 1:size(hdivotes, 1)
		if length(hdivotes.abbr[i]) > 0
			annotations!(hdivotes.abbr[i],
				position = (hdivotes.income[i], hdivotes.hdi[i]),
				textsize = 10)
		end
	end
	hdivotes.rank_hdi = sortperm(hdivotes.hdi)
	global hdivotes2 = sort(hdivotes, :income)
	ax = Axis(f[1, 2]; title = "Ranked HDI ~ ranked income",
		xlabel = "Rank of average state income in 2000",
		ylabel = "Rank of Human Development Index")
	for i in 1:size(hdivotes2, 1)
		if length(hdivotes2.abbr[i]) > 0
			annotations!(hdivotes2.abbr[i],
				position = (i, hdivotes2.rank_hdi[i]),
				textsize = 10)
		end
	end
	current_figure()
end

# ╔═╡ 1cebb5eb-0ae1-4ee4-bf0f-7695b653f721
hdivotes2

# ╔═╡ a2d1e520-977d-46ef-8bb3-5230ade6a2ee
@model function ppl2_1(inc, hdi)
    a ~ Normal(0, 5)
    b ~ Normal(0, 5)
    σ ~ Exponential(1)
    μ = a .+ b .* inc
    for i in eachindex(hdi)
        hdi[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ e0d145a9-3254-4b6f-9a9e-25716dadf390
begin
	rank_income = collect(1:size(hdivotes2, 1))
	rank_hdi = hdivotes2.rank_hdi
	m2_1t = ppl2_1(rank_income, rank_hdi)
	chns2_1t = sample(m2_1t, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns2_1t)
end

# ╔═╡ 74be1757-6b46-49cb-964c-35b6ca022ad0
begin
	post2_1t = DataFrame(chns2_1t)
	ms2_1t = model_summary(post2_1t, [:a, :b, :σ])
end

# ╔═╡ 79a15820-de0f-426d-9c83-2e4113750ca8
ā, b̄, σ̄ = [ms2_1t[p, :median] for p in [:a, :b, :σ]]

# ╔═╡ 1bcc7dde-85ef-48b7-b7ff-9d9560bdc16d
let
	f = Figure()
	ax = Axis(f[1, 1]; title = "HDI ~ income",
		xlabel = "Average state income in 2000",
		ylabel = "Human Development Index")
	for i in 1:size(hdivotes, 1)
		if length(hdivotes.abbr[i]) > 0
			annotations!(hdivotes.abbr[i],
				position = (hdivotes.income[i], hdivotes.hdi[i]),
				textsize = 10)
		end
	end
	ax = Axis(f[1, 2]; title = "Ranked HDI ~ ranked income",
		xlabel = "Rank of average state income in 2000",
		ylabel = "Rank of Human Development Index")
	for i in 1:size(hdivotes2, 1)
		if length(hdivotes2.abbr[i]) > 0
			annotations!(hdivotes2.abbr[i],
				position = (i, hdivotes2.rank_hdi[i]),
				textsize = 10)
		end
	end
	x = 0:52
	lines!(x, ā .+ b̄ .* x; color=:red)
	f
end

# ╔═╡ b95f75e1-5eb8-4862-9f52-1adebc4ec533
md" ### 2.2 Validity and reliability."

# ╔═╡ c21d2adf-d210-4f5d-8d1f-6a0c83d8679d
begin
	pew_pre_raw = CSV.read(ros_datadir("Pew", "pew.csv"), DataFrame; missingstring="NA", pool=false)
	pew_pre = pew_pre_raw[:, [:survey, :regicert,  :party, :state, :heat2, :heat4, :income2, :party4, :date,
		:weight, :voter_weight2, :pid, :ideology, :inc]]
end

# ╔═╡ add96785-f500-4981-a552-87cde06677e6
pid_incprob = CSV.read(ros_datadir("Pew", "pid_incprop.csv"), DataFrame; missingstring="NA", pool=false)

# ╔═╡ cd09e1ca-0b6c-437c-9eba-c571347a0832
ideo_incprob = CSV.read(ros_datadir("Pew", "ideo_incprop.csv"), DataFrame; missingstring="NA", pool=false)

# ╔═╡ 68ffb634-7521-44ab-804b-b92a603d6aae
begin
	party_incprob_df = CSV.read(ros_datadir("Pew", "party_incprop.csv"), DataFrame; missingstring="NA", pool=false)
	party_incprob = reshape(Array(party_incprob_df)[:, 2:end], :, 3, 9)
	party_incprob[:, :, 9]
end

# ╔═╡ f19cabdb-3af5-4ab7-981e-601c15e9829b
let
	x1 = 1.0:1.0:9.0
	f = Figure()
	ax = Axis(f[1, 1], title = "Self-declared political ideology by income",
		xlabel = "Income category", ylabel = "Vote fraction")
	limits!(ax, 1, 9, 0, 1)
	for i in 1:6
		sca1 = scatter!(x1, Array(ideo_incprob[i, 2:end]))
		lin = lines!(x1, Array(ideo_incprob[i, 2:end]))
		band!(x1, fill(0, length(x1)), Array(ideo_incprob[i, 2:end]);
			color = (:blue, 0.25), label = "Label")
	end
	annotations!("Very conservative", position = (3.2, 0.945), textsize=15)
	annotations!("Conservative", position = (3.9, 0.78), textsize=15)
	annotations!("Moderate", position = (4.0, 0.4), textsize=15)
	annotations!("Liberal", position = (4.2, 0.1), textsize=15)
	annotations!("Very liberal", position = (3.8, 0.0075), textsize=15)
	ax = Axis(f[1, 2], title = "Self-declared party indentification by income..",
		xlabel = "Income category", ylabel = "Vote fraction")
	limits!(ax, 1, 9, 0, 1)
	for i in 1:6
		sca1 = scatter!(x1, Array(pid_incprob[i, 2:end]))
		lin = lines!(x1, Array(pid_incprob[i, 2:end]))
		band!(x1, fill(0, length(x1)), Array(pid_incprob[i, 2:end]);
			color = (:blue, 0.25), label = "Label")
	end
	annotations!("Republican", position = (4.0, 0.87), textsize=15)
	annotations!("Lean Rep", position = (4.15, 0.675), textsize=15)
	annotations!("Independent", position = (3.95, 0.53), textsize=15)
	annotations!("Lean Dem", position = (4.2, 0.4), textsize=15)
	annotations!("Democrat", position = (4.1, 0.19), textsize=15)
	current_figure()
end

# ╔═╡ 20b7fa5b-0272-443f-afee-e945584d8bfc
md" ### 2.3 All graphs are comparisons."

# ╔═╡ 3a5363ed-e6c9-4c85-9c6e-8f74e99be6be
health = CSV.read(ros_datadir("HealthExpenditure", "healthdata.csv"), DataFrame; missingstring="NA", pool=false)

# ╔═╡ 561661dd-732d-4f7e-acf5-1c8d563cb51a
@model function ppl2_3(x, y)
    a ~ Normal(0, 50)
    b ~ Normal(0, 50)
    σ ~ Exponential(1)
    μ = a .+ b .* x
    for i in eachindex(y)
        y[i] ~ Normal(μ[i], σ)
    end
end

# ╔═╡ 85f51724-2b0b-42b5-80b0-3a9cc2060c4b
begin
	m2_3t = ppl2_3(health.spending, health.lifespan)
	chns2_3t = sample(m2_3t, NUTS(), MCMCThreads(), 1000, 4)
	describe(chns2_3t)
end

# ╔═╡ 07e30525-e421-4905-9ed6-0f0b06c5e2c1
begin
	post2_3t = DataFrame(chns2_3t)[:, 3:5]
	ms2_3t = model_summary(post2_3t, [:a, :b, :σ])
end

# ╔═╡ a753917e-e88b-49c7-b65e-865f7109eae5
â, b̂, σ̂ = ms2_3t[:, :median]

# ╔═╡ 4dc4adec-a059-4332-81fd-e77a675eb40b
let
	x = 0:8000
	f = Figure()
	ax = Axis(f[1, 1], title = "Health expenditure of 30 `western` countries",
		xlabel = "Spending [PPP US\$]", ylabel = "Life expectancy [years]")
	limits!(ax, 0, 8100, 73, 83)
	sca = scatter!(health.spending, health.lifespan; color=:darkred)
	lin = lines!(x, â .+ b̂ * x; color=:lightblue)
	for i in 1:nrow(health)
		if health.country[i] == "UK"
			annotations!(health.country[i], position = (health.spending[i]+40, health.lifespan[i]-0.25), textsize=8)
		elseif health.country[i] == "Finland"
			annotations!(health.country[i], position = (health.spending[i]-100, health.lifespan[i]+0.1), textsize=8)
		elseif health.country[i] == "Greece"
			annotations!(health.country[i], position = (health.spending[i]-300, health.lifespan[i]-0.25), textsize=8)
		elseif health.country[i] == "Sweden"
			annotations!(health.country[i], position = (health.spending[i]-180, health.lifespan[i]-0.25), textsize=8)
		elseif health.country[i] == "Ireland"
			annotations!(health.country[i], position = (health.spending[i]-150, health.lifespan[i]-0.25), textsize=8)
		elseif health.country[i] == "Netherlands"
			annotations!(health.country[i], position = (health.spending[i]+50, health.lifespan[i]+0.01), textsize=8)
		elseif health.country[i] == "Germany"
			annotations!(health.country[i], position = (health.spending[i]-350, health.lifespan[i]+0.08), textsize=8)
		elseif health.country[i] == "Austria"
			annotations!(health.country[i], position = (health.spending[i]+30, health.lifespan[i]-0.2), textsize=8)
		else
			annotations!(health.country[i], position = (health.spending[i]+60, health.lifespan[i]-0.1), textsize=8)
		end
	end
	current_figure()
end

# ╔═╡ caf0483a-eab8-419a-b04a-57fce2c261a8
md" #### Names example."

# ╔═╡ 149ad128-8194-48ac-90ce-b7d09f2c1272
cleannames = CSV.read(ros_datadir("Names", "allnames_clean.csv"), DataFrame)

# ╔═╡ 08306ca7-8411-410a-a1d7-8e9cf7ed9c4d
size(cleannames)

# ╔═╡ 3ae0c23a-c447-483f-b580-95fec4b6b922
names(cleannames)

# ╔═╡ 86ce4828-82aa-43a4-809f-4172090166ee
df = cleannames[cleannames.sex .== "M", ["name", "sex", "X1906", "X1956", "X2006"]]

# ╔═╡ 4e743208-1a83-49cc-bfbf-70e2d0fb4103
letters = 'a':'z';

# ╔═╡ 649b44be-4243-4a43-b0ae-ccdb6f10b4df
function count_letters(df::DataFrame, years::Vector{String})
	letter_counts = DataFrame()
	for year in Symbol.(years)
		
		!(year in Symbol.(names(df))) && begin
			@warn "The year $(year) is not present in df."
			continue
		end
		
		tmpdf = df[:, [:name, year]]
		
		yrcounts = zeros(Int, length(letters))
		for (ind, letter) in enumerate(letters)
			yrcounts[ind] = sum(filter(row -> row.name[end] == letter, tmpdf)[:, 2])
		end
		letter_counts[!, year] = 100 * yrcounts / sum(yrcounts)
	end
	letter_counts
end

# ╔═╡ bfe5d5ae-f335-4ef2-ba77-16047a527211
letter_count = count_letters(df, ["X1906", "X1956", "X2006"])

# ╔═╡ b48058d6-a7bd-4e35-9382-4b89172b612b
sum.(eachcol(letter_count))

# ╔═╡ ed3e4a1a-986a-48e3-b333-9d9cd5f03591
let
	f = Figure()
	for (ind, year) in enumerate(["X1906", "X1956", "X2006"])
		ax = Axis(f[ind, 1], title="Last letters in boys' names in $(year[2:end])",
			ylabel="Perc of names")
		ax.xticks = (0:27, vcat(" ", string.(letters), " "))
		barplot!(f[ind, 1], 1:26, letter_count[:, Symbol(year)], width=0.8, gap=0.01)
	end
	f
end

# ╔═╡ 738c9b8c-1b2c-4f70-92c5-fe8fb1e252cc
all_letter_count = count_letters(cleannames[cleannames.sex .== "M", :], names(cleannames[:, vcat(4:end)]))

# ╔═╡ 4ef9003a-84fa-4b2f-baa6-d43fd059ed89
all_letter_count[:, "X1906"]

# ╔═╡ 243ee045-12b7-40e5-a1a6-3a0a598d7cdd
let
	f = Figure()
	ax = Axis(f[1, 1], title="Last letters in boys' names over time",
		ylabel="Perc of all boys' names that year")
	ax.xticks = (6:25:131, ["1886", "1906", "1931", "1956", "1981", "2006"])

	for l in 1:length(letters)
		col = :lightgrey
		if letters[l] == 'n'
			col = :darkblue
		elseif letters[l] == 'd'
			col = :darkred
		elseif letters[l] == 'y'
			col = :darkgreen
		end
		if maximum(Array(all_letter_count)[l,:]) > 1
			lines!(1:size(all_letter_count, 2), Array(all_letter_count)[l,:], color=col)
		end
		annotations!("n", position = (106, 25), textsize=15)
		annotations!("d", position = (56, 18), textsize=15)
		annotations!("y", position = (106, 11), textsize=15)

	end
	current_figure()
end

# ╔═╡ 6b5e6ea1-a66f-4c4b-b220-ebc05f96ea5e
md" ### 2.4 Data and adjustment."

# ╔═╡ fb925b13-efa2-4d77-8e75-e00cfd294902
md" #### Not yet done. A good exercise ...?"

# ╔═╡ 4e009883-b9ca-4001-a8c2-748b096bc0b1
age = CSV.read(ros_datadir("AgePeriodCohort", "US-EST00INT-ALLDATA.csv"), DataFrame)

# ╔═╡ 62b481f7-b52d-4bd5-be07-f2aaea9ab985
names(age)

# ╔═╡ d6cda2f8-af2e-4079-8534-4c7ad13da79b
white_nonhisp = CSV.read(ros_datadir("AgePeriodCohort", "white_nonhisp_death_rates.txt"), DataFrame; delim="\t")

# ╔═╡ 856b8105-68d3-4b8a-bf28-d164b94b0946
white_hisp = CSV.read(ros_datadir("AgePeriodCohort", "white_nonhisp_death_rates_by_sex.txt"), DataFrame; delim="\t")

# ╔═╡ 6ccd2583-5983-4586-b653-2f5bbe09f77d
white_nonhisp_by_sex = CSV.read(joinpath(expanduser("~"), ".julia", "dev", "RegressionAndOtherStories", "data", "AgePeriodCohort", "white_nonhisp_death_rates_by_sex.txt"), DataFrame; delim="\t")

# ╔═╡ Cell order:
# ╟─eb7ea04a-da52-4e69-ac3e-87dc7f014652
# ╟─d7543b63-52d3-449b-8ce3-d979c23f8b95
# ╠═ed172871-fa4d-4111-ac0a-341898917948
# ╠═5084b8f0-65ac-4704-b1fc-2a9008132bd7
# ╟─4755dab0-d228-41d3-934a-56f2863a5652
# ╠═f71640c9-3918-475e-b32b-c85424bbcf5e
# ╟─ab84df44-ddde-4b7e-9370-b22646fe0a3b
# ╠═6c2043f4-dcbd-4535-a5ae-61c109519879
# ╠═d7b5bfec-3ca7-46fe-aed5-094ee266016b
# ╠═2dd08df7-a6dd-4a19-a673-a98c43e4d552
# ╠═c4d6df3e-3cc9-4588-84e6-fe7265f01027
# ╠═696eb9b0-f239-425b-8b71-802dfd9d8a42
# ╠═1cebb5eb-0ae1-4ee4-bf0f-7695b653f721
# ╠═a2d1e520-977d-46ef-8bb3-5230ade6a2ee
# ╠═e0d145a9-3254-4b6f-9a9e-25716dadf390
# ╠═74be1757-6b46-49cb-964c-35b6ca022ad0
# ╠═79a15820-de0f-426d-9c83-2e4113750ca8
# ╠═1bcc7dde-85ef-48b7-b7ff-9d9560bdc16d
# ╟─b95f75e1-5eb8-4862-9f52-1adebc4ec533
# ╠═c21d2adf-d210-4f5d-8d1f-6a0c83d8679d
# ╠═add96785-f500-4981-a552-87cde06677e6
# ╠═cd09e1ca-0b6c-437c-9eba-c571347a0832
# ╠═68ffb634-7521-44ab-804b-b92a603d6aae
# ╠═f19cabdb-3af5-4ab7-981e-601c15e9829b
# ╟─20b7fa5b-0272-443f-afee-e945584d8bfc
# ╠═3a5363ed-e6c9-4c85-9c6e-8f74e99be6be
# ╠═561661dd-732d-4f7e-acf5-1c8d563cb51a
# ╠═85f51724-2b0b-42b5-80b0-3a9cc2060c4b
# ╠═07e30525-e421-4905-9ed6-0f0b06c5e2c1
# ╠═a753917e-e88b-49c7-b65e-865f7109eae5
# ╠═4dc4adec-a059-4332-81fd-e77a675eb40b
# ╟─caf0483a-eab8-419a-b04a-57fce2c261a8
# ╠═149ad128-8194-48ac-90ce-b7d09f2c1272
# ╠═08306ca7-8411-410a-a1d7-8e9cf7ed9c4d
# ╠═3ae0c23a-c447-483f-b580-95fec4b6b922
# ╠═86ce4828-82aa-43a4-809f-4172090166ee
# ╠═4e743208-1a83-49cc-bfbf-70e2d0fb4103
# ╠═649b44be-4243-4a43-b0ae-ccdb6f10b4df
# ╠═bfe5d5ae-f335-4ef2-ba77-16047a527211
# ╠═b48058d6-a7bd-4e35-9382-4b89172b612b
# ╠═ed3e4a1a-986a-48e3-b333-9d9cd5f03591
# ╠═738c9b8c-1b2c-4f70-92c5-fe8fb1e252cc
# ╠═4ef9003a-84fa-4b2f-baa6-d43fd059ed89
# ╠═243ee045-12b7-40e5-a1a6-3a0a598d7cdd
# ╟─6b5e6ea1-a66f-4c4b-b220-ebc05f96ea5e
# ╟─fb925b13-efa2-4d77-8e75-e00cfd294902
# ╠═4e009883-b9ca-4001-a8c2-748b096bc0b1
# ╠═62b481f7-b52d-4bd5-be07-f2aaea9ab985
# ╠═d6cda2f8-af2e-4079-8534-4c7ad13da79b
# ╠═856b8105-68d3-4b8a-bf28-d164b94b0946
# ╠═6ccd2583-5983-4586-b653-2f5bbe09f77d
