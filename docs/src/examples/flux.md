# Usage example with FLux.jl

[Flux.jl](http://lux.csail.mit.edu/stable/) is one of the defacto deep learning framework
in Julia.

## Step 0: Import the necessary packages

!!! tip

    You can install these packages using
    `using Pkg; Pkg.add.(["CUDA", "Optimisers", "Flux", "Random", "Zygote"])`

```julia
using CUDA, Optimisers, FluxMPI, Flux, Random, Zygote
```

## Step 1: Initialize FluxMPI. Not doing this will segfault your code

```julia
FluxMPI.Init()
CUDA.allowscalar(false)
```

## Step 2: Sync Model Parameters and States

```julia
model = Chain(Dense(1 => 256, tanh), Dense(256 => 512, tanh), Dense(512 => 256, tanh),
              Dense(256 => 1))

rng = Random.default_rng()
Random.seed!(rng, local_rank())
# Always remember to wrap the model in FluxMPIFluxModel
model = FluxMPI.synchronize!(FluxMPIFluxModel(model); root_rank=0)
```

## Step 3: Ensure data is properly partitioned

It is the user's responsibility to partition the data across the processes. In this case,
we are training on a total of `16 * <np>` samples. Instead of manually partitioning the
data, we can use [`DistributedDataContainer`](@ref) to partition the data.

```julia
x = rand(rng, 1, 16) |> gpu
y = x .^ 2
```

## Step 4: Wrap the optimizer in [`DistributedOptimizer`](@ref)

Remember to scale the learning rate by the number of workers [`total_workers`](@ref).

```julia
opt = DistributedOptimizer(Optimisers.Adam(0.001f0))
st_opt = Optimisers.setup(opt, model)

loss(model) = sum(abs2, model(x) .- y)
```

## Step 5: Synchronize the optimizer state

```julia
st_opt = FluxMPI.synchronize!(st_opt; root_rank = 0)
```

## Step 6: Train your model

Remember to print using [`fluxmpi_println`](@ref) or [`fluxmpi_print`](@ref).


```julia
gs_ = gradient(loss, model)[1]
Optimisers.update(st_opt, ps, gs_)

t1 = time()

for epoch in 1:100
  global model, st_opt
  l, back = Zygote.pullback(loss, model)
  FluxMPI.fluxmpi_println("Epoch $epoch: Loss $l")
  gs = back(one(l))[1]
  st_opt, model = Optimisers.update(st_opt, model, gs)
end

FluxMPI.fluxmpi_println(time() - t1)
```

Run the code using `mpiexecjl -n 3 julia --project=. <filename>.jl`.

## Complete Script

```julia
using CUDA, FluxMPI, Flux, Optimisers, Random, Zygote

FluxMPI.Init()
CUDA.allowscalar(false)

model = Chain(Dense(1 => 256, tanh), Dense(256 => 512, tanh), Dense(512 => 256, tanh),
              Dense(256 => 1)) |> gpu

rng = Random.default_rng()
Random.seed!(rng, local_rank())
model = FluxMPI.synchronize!(FluxMPIFluxModel(model); root_rank=0)

x = rand(rng, 1, 16) |> gpu
y = x .^ 2

opt = DistributedOptimizer(Optimisers.Adam(0.001f0))
st_opt = Optimisers.setup(opt, model)

loss(model) = sum(abs2, model(x) .- y)

st_opt = FluxMPI.synchronize!(st_opt; root_rank = 0)

gs_ = gradient(loss, model)[1]
Optimisers.update(st_opt, ps, gs_)

t1 = time()

for epoch in 1:100
  global model, st_opt
  l, back = Zygote.pullback(loss, model)
  FluxMPI.fluxmpi_println("Epoch $epoch: Loss $l")
  gs = back(one(l))[1]
  st_opt, model = Optimisers.update(st_opt, model, gs)
end

FluxMPI.fluxmpi_println(time() - t1)
```