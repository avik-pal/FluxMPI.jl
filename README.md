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
using Flux, FluxMPI, MPI, Zygote, CUDA

FluxMPI.Init()
CUDA.allowscalar(false)

total_gpus = length(CUDA.devices())
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

model = Chain(Dense(1, 2, tanh), Dense(2, 1))

model_dp = DataParallelFluxModel(model)

ps = Flux.params(model_dp)

x = rand(1, 64) |> gpu
y = x .^ 2

dataloader = DataParallelDataLoader((x, y), batchsize = 16)

function loss(x_, y_)
    loss = sum(abs2, model_dp(x_) .- y_)
    println("Process [$rank / $size]: Loss = $loss")
    return loss
end

for epoch in 1:100
    if rank == 0
        @info "epoch = $epoch" 
    end
    Flux.Optimise.train!(loss, ps, dataloader, Flux.ADAM(0.001))
end
```

Run the code using `mpiexecjl -n 3 julia --project=. <filename>.jl`.

## Usage Instructions

1. Call `FluxMPI.Init()`
2. Wrap your model in `DataParallelFluxModel`
3. Use `DataParallelDataLoader` instead of `Flux.Data.DataLoader`. If you are using a custom DataLoader you need to ensure that the data is split appropriately.
4. Modify logging code. You don't want to log from all processes. Instead just log from `rank == 0`.
5. `Zygote.pullback` is not overloaded. Use `Zygote.withgradient` or `Zygote.gradient` instead.
6. If any of your functions dispatch on `typeof(model)` then you need to define a dispatch for `dp::DataParallelFluxModel` and call the function using `dp.model`.

## API Reference

* `FluxMPI.Init(; gpu_devices = nothing)`

  * `gpu_devices`: List of GPU Devices to use. If `nothing` and `CUDA.functional() == true` then use all available GPUs in a round robin fashion.

* `DataParallelFluxModel`:

  * `model`: Any Flux model.

* `DataParallelParamsWrapper`: Returned by `Flux.params(::DataParallelFluxModel)`. Behaves identical to `Zygote.Params` but synchronizes the gradients for each `Zygote.gradient` call.

* `DataPrallelDataLoader`: Wrapper around `DataLoader` and ensures a proper split of data between the different processes.
