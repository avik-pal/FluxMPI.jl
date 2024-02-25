# FluxMPI.jl

> [!CAUTION]
> This package should be considered deprecated and won't receive any updates. Distributed Training will become a native feature for Lux, so it makes little sense for me to maintain an additional package that does the same thing. Track https://github.com/LuxDL/Lux.jl/issues/494 for furthur updates.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://avik-pal.github.io/FluxMPI.jl/stable/)
[![Latest](https://img.shields.io/badge/docs-dev-blue.svg)](https://avik-pal.github.io/FluxMPI.jl/dev/)

[![CI](https://github.com/avik-pal/FluxMPI.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/avik-pal/FluxMPI.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/avik-pal/FluxMPI.jl/branch/main/graph/badge.svg?token=1L3ePmqyPo)](https://codecov.io/github/avik-pal/FluxMPI.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/FluxMPI)](https://pkgs.genieframework.com?packages=FluxMPI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Distributed Data Parallel Training of Neural Networks

## Installation

Stable release:

```julia
] add FluxMPI
```

Latest development version:

```julia
] add FluxMPI#main
```

## Quick Start

```julia
using CUDA, FluxMPI, Lux, Optimisers, Random, Zygote

FluxMPI.Init()
CUDA.allowscalar(false)

model = Chain(Dense(1 => 256, tanh), Dense(256 => 512, tanh), Dense(512 => 256, tanh),
              Dense(256 => 1))
rng = Random.default_rng()
Random.seed!(rng, local_rank())
ps, st = Lux.setup(rng, model) .|> gpu

ps = FluxMPI.synchronize!(ps; root_rank = 0)
st = FluxMPI.synchronize!(st; root_rank = 0)

x = rand(rng, 1, 16) |> gpu
y = x .^ 2

opt = DistributedOptimizer(Adam(0.001f0))
st_opt = Optimisers.setup(opt, ps)

loss(p) = sum(abs2, model(x, p, st)[1] .- y)

st_opt = FluxMPI.synchronize!(st_opt; root_rank = 0)

gs_ = gradient(loss, ps)[1]
Optimisers.update(st_opt, ps, gs_)

t1 = time()

for epoch in 1:100
  global ps, st_opt
  l, back = Zygote.pullback(loss, ps)
  FluxMPI.fluxmpi_println("Epoch $epoch: Loss $l")
  gs = back(one(l))[1]
  st_opt, ps = Optimisers.update(st_opt, ps, gs)
end

FluxMPI.fluxmpi_println(time() - t1)
```

Run the code using `mpiexecjl -n 3 julia --project=. <filename>.jl`.

## Examples

* [Deep Equilibrium Models](https://github.com/SciML/FastDEQ.jl) -- Deep Implicit Neural
  Networks & Infinite Time Neural ODEs
* [ImageNet Training with Lux.jl](https://github.com/avik-pal/Lux.jl/tree/main/examples/ImageNet)

## Style Guide

We follow the [Lux Style Guide](http://lux.csail.mit.edu/stable/devdocs/style_guide/). All
contributions must adhere to this style guide.

## Changelog

### v0.7

* Dropped support for MPI v0.19.
* `FLUXMPI_DISABLE_CUDAMPI_SUPPORT` is no longer used. Instead use
  `FluxMPI.disable_cudampi_support()` to setup a LocalPreferences.toml file.
* `clean_(print/println)` functions are now `fluxmpi_(print/println)`.

<details>

<summary><h3>v0.6</h3></summary>

* Dropped support for `LearnBase`, aka `DataLoaders.jl`. `DistributedDataContainer` is now
  the only compatible with `MLUtils.jl`.
* `DistributedOptimiser` name changed to `DistributedOptimizer`.

</details>

<details>

<summary><h3>v0.5</h3></summary>

#### v0.5.3

* Introduces a new API for gradient synchronization
  * Don't wrap in `DistributedOptimiser`
  * Instead just add a line `allreduce_gradients(gs::NamedTuple)`

#### v0.5.1

* Internal `MPIExtensions` functions renamed
  * `Allreduce!` --> `allreduce!`
  * `Bcast!` --> `bcast!`
  * `Reduce!` --> `reduce!`
* CUDA-unaware MPI bug resolved https://github.com/avik-pal/Lux.jl/issues/18
* Disable CUDA-aware MPI support from `FluxMPI` using `FLUXMPI_DISABLE_CUDAMPI_SUPPORT=true`
* Temporarily re-added dependencies on `MLDataUtils` and `LearnBase` to ensure
  `DataLoaders.jl` still works -- This will be dropped in a future release

#### v0.5.0

* `DistributedOptimiser` no longer averages the gradients. Instead, the values are summed
  across the processes. To ensure averaging divide the loss by `total_workers()`
* `rrule`s and `frule`s defined for `local_rank()` and `total_workers` -- they can now be
  safely used inside loss functions.

</details>

<details>

<summary><h3>v0.4</h3></summary>

* `fluxmpi_print` and `fluxmpi_println` print the current time even if `FluxMPI` has not been
  initialized.
* Calling `local_rank` or `total_workers` before `FluxMPI.Init` doesn't lead to a segfault.
  Rather we throw an error.
* `MLDataUtils` and `LearnBase` dependencies have been dropped
  (See https://github.com/avik-pal/FluxMPI.jl/issues/17)
* `Zygote` and `Flux` dependencies have been removed
    * No dispatch for `FluxMPI.synchronize!` is now available for `Zygote.Params`. Instead
      users should be manually broadcasting the function over `Zygote.Params`

</details>

<details>

<summary><h3>v0.3</h3></summary>

* `broadcast_parameters` has been renamed to `FluxMPI.synchronize!` since it synchronizes
  a lot more than trainable parameters now.
* DistributedOptimiser is no longer tied with Flux. We can essentially deal with any
  training as long as it is compatible with
  [Optimisers.jl](https://github.com/FluxML/Optimisers.jl)

</details>
