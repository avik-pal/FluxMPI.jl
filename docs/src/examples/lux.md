# Usage example with Lux.jl

[Lux.jl](http://lux.csail.mit.edu/stable/) is a deep learning framework which provides a
functional design to implement neural networks in Julia. If you are not familiar with Lux,
first check-out
[this tutorial](http://lux.csail.mit.edu/stable/examples/generated/beginner/Basics/main/)
before proceeding.

## Step 0: Import the necessary packages

!!! tip

    You can install these packages using
    `using Pkg; Pkg.add.(["CUDA", "Optimisers", "Lux", "Random", "Zygote"])`

```julia
using CUDA, Optimisers, FluxMPI, Lux, Random, Zygote
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
ps, st = Lux.setup(rng, model) .|> gpu

ps = FluxMPI.synchronize!(ps; root_rank = 0)
st = FluxMPI.synchronize!(st; root_rank = 0)
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
opt = DistributedOptimizer(Adam(0.001f0))
st_opt = Optimisers.setup(opt, ps)

loss(p) = sum(abs2, model(x, p, st)[1] .- y)
```

## Step 5: Synchronize the optimizer state

```julia
st_opt = FluxMPI.synchronize!(st_opt; root_rank = 0)
```

## Step 6: Train your model

Remember to print using [`fluxmpi_println`](@ref) or [`fluxmpi_print`](@ref).


```julia
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


## Complete Script

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