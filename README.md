# FluxMPI.jl

Data Parallel Training of Flux Models.

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
using Flux, FluxMPI, CUDA, Optimisers

# Step 1: Initialize FluxMPI. Not doing this will segfault your code
FluxMPI.Init()
CUDA.allowscalar(false)

# Step 2: Sync Model Parameters
model = Chain(Dense(1, 2, tanh), Dense(2, 1)) |> gpu
FluxMPI.synchronize!(model; root_rank = 0)

# It is the user's responsibility to partition the data across the processes
# In this case, we are training on a total of 16 * <np> samples
x = rand(Float32, 1, 16) |> gpu
y = x .^ 2
dataloader = Flux.DataLoader((x, y), batchsize = 16)

# Step 3: Wrap the optimizer in DistributedOptimizer
#         Scale the learning rate by the number of workers (`total_workers()`).
opt = DistributedOptimiser(Optimisers.ADAM(0.001f0))
st = Optimisers.setup(opt, model)

# Step 4: synchronize! the optimizer state
FluxMPI.synchronize!(st; root_rank = 0)

loss(x_, y_) = sum(abs2, model(x_) .- y_)

for epoch in 1:100
    local_rank() == 0 && @info "epoch = $epoch"

    for (x_, y_) in dataloader
        gs = gradient(model) do model
            sum(abs2, model(x_) .- y_)
        end
        st, model = Optimisers.update!(st, model, gs)
    end
end
```

Run the code using `mpiexecjl -n 3 julia --project=. <filename>.jl`.

## Examples

* [Deep Equilibrium Models](https://github.com/SciML/FastDEQ.jl) -- Deep Implicit Neural Networks & Infinite Time Neural ODEs
  * [Deep Equilibrium Models Paper](https://arxiv.org/abs/1909.01377)
  * [MultiScale Deep Equilibrium Models Paper](https://arxiv.org/abs/2006.08656)
  * [Infinite Time Neural ODE Paper](https://arxiv.org/abs/2201.12240)
* [Image Classification using FastAI.jl](/examples/fastai/train.jl): Only a few changes are needed for the integration
  * Install `FluxTraining#ap/improvements` -- I am happy to upstream the changes but they are a bit opinionated...
  * Remember to do `FluxMPI.Init()`
  * Convert data -> `data = DistributedDataContainer(data)`
  * Finally pass in the optimizer `optimizer = DistributedOptimiser(ADAM())`
  * Start the code with `mpiexecjl -n <np> julia --project=examples/fastai/ train.jl`

## Usage Instructions

There are essentially 6 main steps to remember:

1. Initialize FluxMPI (`FluxMPI.Init()`)
2. Sync Model Parameters (`synchronize!(Flux.params(model); root_rank)`)
3. Dealing with DataLoading. There are two options:
   1. Manually distribute the data across the processes. If all the processes are using the same data, it becomes quite pointless
   2. Use `DistributedDataContainer`. It takes the `data` and splits it evenly across all the processes. The only assumption is that the `data` is compatible with [LearnBase.jl](https://github.com/JuliaML/LearnBase.jl) API. The returned container is compatible with [LearnBase.jl](https://github.com/JuliaML/LearnBase.jl) so [DataLoaders.jl](https://lorenzoh.github.io/DataLoaders.jl/dev/) should work by default.
4. Wrap Optimizer in `DistributedOptimizer`
5. Sync the optimizer state across the processes
6. Change logging code to check for `local_rank() == 0`

Finally, start the code using `mpiexecjl -n <np> julia --project=. <filename>.jl`

## API Reference

All functions have dedicated docstrings. Use the help mode in REPL to access them

### MPIExtensions

**NOTE: Functions are not exported**

1. `Reduce!`
2. `Allreduce!`
3. `Bcast!`
4. `Iallreduce!`

### FluxMPI

1. `FluxMPI.Init` (**not exported since name is very common**)
2. `DistributedOptimiser`
3. `FluxMPI.synchronize!` (**not exported since name is very common**)
4. `DistributedDataContainer`

## Changelog

### v0.3

* `broadcast_parameters` has been renamed to `FluxMPI.synchronize!` since it synchronize!s a lot more than trainable parameters now.
* DistributedOptimiser is no longer tied with Flux. We can essentially deal with any training as long as it is compatible with [Optimisers.jl](https://github.com/FluxML/Optimisers.jl)

## Known Caveats

1. `Iallreduce!` uses `@async` when using CuArrays. The other alternative right now is to hit a segfault
2. Using any form of MPI syncing operation like `MPI.Barrier()` without waiting for Julia tasks (see point 1) to finish can lead to deadlocks

## Other Data Parallel Training Libraries

* [ResNetImageNet.jl](https://github.com/DhairyaLGandhi/ResNetImageNet.jl)
