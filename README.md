# ROSTuringPluto

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->

## Note

After many years I have decided to step away from my work with Stan and Julia. My plan is to be around until the end of 2024 for support if someone decides to step in and take over further development and maintenance work.

At the end of 2024 I'll archive the different packages and projects included in the Github organisations StanJulia, StatisticalRethingJulia and RegressionAndOtherStoriesJulia if no one is interested (and time-wise able!) to take on this work.

I have thoroughly enjoyed working on both Julia and Stan and see both projects mature during the last 15 or so years. And I will always be grateful for the many folks who have helped me on numerous occasions. Both the Julia and the Stan community are awesome to work with! Thanks a lot!

## Purpose

This project will contain (work is in early stages of progress!) a set of Pluto notebooks that contain Julia versions of the examples in the R project `ROS-Examples` based on the book ["Regression and Other Stories" by A Gelman, J Hill and A Vehtari](https://www.cambridge.org/highereducation/books/regression-and-other-stories/DD20DD6C9057118581076E54E40C372C#overview).

These notebooks are intended to be used in conjunction with above book.

Each notebook contains a chapter.  In the `pdfs` directory there are also PDF versions of the chapters created with PlutoPDF (see notes.md).

## Personal note

This project will take quite a while to complete, I expect at least a year. But it has a special meaning to me: When I started to work on Julia interfaces for Stan's cmdstan binary in 2011, I did that to work through the ["ARM" book](http://www.stat.columbia.edu/~gelman/arm/). The ["ROS" book](https://www.cambridge.org/highereducation/books/regression-and-other-stories/DD20DD6C9057118581076E54E40C372C#overview) in a sense the successor to the ARM book.

## Prerequisites

1. A functioning [Julia](https://julialang.org/downloads/).
2. A minimal Julia base environment containing `Pkg` and `Pluto`.

## Setup the Pluto based ROSTuringPluto notebooks

To (locally) use this project, do the following:

Select and download ROSTuringPluto.jl from [RegressionAndOtherStoriesJulia](https://github.com/RegressionAndOtherStoriesJulia/), e.g. to clone it to the `~/.julia/dev/ROSTuringPluto` directory:

```Julia
$ cd ~/.julia/dev
$ git clone https://github.com/RegressionAndOtherStoriesJulia/ROSTuringPluto.jl ROSTuringPluto
$ cd ROSTuringPluto/notebooks # Move to the (downloaded) notebook directory
$ julia # Start Julia REPL
```

Still in the Julia REPL, start a Pluto notebook server.
```Julia
julia> using Pluto
julia> Pluto.run()
```

A Pluto page should open in a browser. See [this page](https://www.juliafordatascience.com/first-steps-5-pluto/) for a quick Pluto introduction.

Select a notebook in the `open a file` entry box, e.g. type `./` and select `chapters` Select a chapter notebook and press `open`.
