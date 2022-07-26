md"#### See chapter 6 in Regression and Other Stories."


md" ##### Widen the notebook."

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

using Pkg, DrWatson

md"##### A typical set of Julia packages to include in notebooks."

begin
	# Specific to this notebook
    using GLM

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

md"### X.1 ..."

hdi = CSV.read(ros_datadir("HDI", "hdi.csv"), DataFrame)
