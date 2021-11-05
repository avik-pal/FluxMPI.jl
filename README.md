# FluxMPI.jl

Project Status: **Experimental**

Data Parallel Training of Flux Models. This is mostly an experimental package that I am using for my research projects.

## Quick Start

```julia
using Flux, FluxMPI, MPI, Zygote, CUDA

MPI.Init()
CUDA.allowscalar(false)

total_gpus = length(CUDA.devices())
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

model = Chain(Dense(1, 2, tanh), Dense(2, 1))

model_dp = DataParallelFluxModel(
    model,
    [i % total_gpus for i = 1:MPI.Comm_size(MPI.COMM_WORLD)],
)

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

1. Wrap your model in `DataParallelFluxModel`
2. Use `DataParallelDataLoader` instead of `Flux.Data.DataLoader`. If you are using a custom DataLoader you need to ensure that the data is split appropriately.
3. Modify logging code. You don't want to log from all processes. Instead just log from `rank == 0`.

## API Reference

* `DataParallelFluxModel`:
  
    * `model`: Any Flux model
    * `gpu_devices`: List of GPU Devices to use. Must be non-empty if GPU is functional on the system.


* `DataParallelParamsWrapper`: Returned by `Flux.params(::DataParallelFluxModel)`. Behaves identical to `Zygote.Params` but synchronizes the gradients for each `Zygote.gradient` call.

* `DataPrallelDataLoader`: Wrapper around `DataLoader` and ensures a proper split of data between the different processes.
