# FluxMPI.jl

Project Status: **Experimental**

Data Parallel Training of Flux Models. This is mostly an experimental package that I am using for my research projects. Additionally, this is primarily geared towards MultiGPU training and not multiple node training.

## Quick Start

```julia
using Flux, FluxMPI, MPI, Zygote

MPI.Init()

model = Chain(Dense(1, 2, tanh), Dense(2, 1))

model_dp = DataParallelFluxModel(model, [0 for _ in 1:MPI.Comm_size(MPI.COMM_WORLD)])

ps = Flux.params(model_dp)

x = rand(1, 10) |> gpu

gs = Zygote.gradient(() -> sum(model_dp(x)), Flux.params(model_dp))
```

Run the code using `mpiexecjl -n 3 julia --project=. <filename>.jl`.


## TODOs

- [ ] Can we check if MPI is CUDA aware?