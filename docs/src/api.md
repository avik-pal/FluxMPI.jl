```@meta
CurrentModule = FluxMPI
```

```@index
Pages = ["api.md"]
```

## Data Helpers

```@docs
DistributedDataContainer
```

## General Functions

```@docs
FluxMPI.Init
FluxMPI.Initialized
fluxmpi_print
fluxmpi_println
local_rank
total_workers
```

## MPIExtensions: Blocking Communication Wrappers

```@docs
FluxMPI.allreduce!
FluxMPI.bcast!
FluxMPI.reduce!
```

## MPIExtensions: Non-Blocking Communication

```@docs
FluxMPI.Iallreduce!
FluxMPI.Ibcast!
```

## Optimization

```@docs
DistributedOptimizer
allreduce_gradients
```

## Synchronization

```@docs
FluxMPI.synchronize!
```

## Configuration

```@docs
FluxMPI.disable_cudampi_support
```
