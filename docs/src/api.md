```@meta
CurrentModule = FluxMPI
```

## Data Helpers

```@docs
FluxMPI.DistributedDataContainer
```

## General Functions

```@docs
FluxMPI.Init
FluxMPI.Initialized
FluxMPI.clean_print
FluxMPI.clean_println
FluxMPI.local_rank
FluxMPI.total_workers
```

## MPIExtensions

### Blocking Communication Wrappers

```@docs
FluxMPI.MPIExtensions.allreduce!
FluxMPI.MPIExtensions.bcast!
FluxMPI.MPIExtensions.reduce!
```

### Non-Blocking Communication

```@docs
FluxMPI.MPIExtensions.Iallreduce!
FluxMPI.MPIExtensions.Ibcast!
```

## Optimization

```@docs
FluxMPI.DistributedOptimizer
FluxMPI.allreduce_gradients
```

## Synchronization

```@docs
FluxMPI.synchronize!
```

## Index

```@index
Pages = ["api.md"]
```