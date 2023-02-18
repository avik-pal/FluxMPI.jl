# FluxMPI.jl

Distributed Data Parallel Training of Neural Networks

!!! note

    This package has very little to do with Flux. FWIW it doesn't even depend on it.
    It can be seamlessly used with [Flux.jl](https://github.com/FluxML/Flux.jl),
    [Lux.jl](https://github.com/avik-pal/Lux.jl), and pretty much any framework which works
    with [Optimisers.jl](https://github.com/FluxML/Optimisers.jl).

## Installation

Install [julia v1.6 or above](https://julialang.org/downloads/). Next Install the stable
release:

```julia
using Pkg
Pkg.add("FluxMPI")
```

To install the Latest development version (not very beneficial we release most patches
almost immediately):

```julia
using Pkg
Pkg.add("FluxMPI", rev="main")
```

## Design Principles

  - Efficient distributed data parallelism using MPI.
  - Not tied to any specific framework -- works with
    [Lux.jl](https://github.com/avik-pal/Lux.jl),
    [Flux.jl](https://github.com/FluxML/Flux.jl), etc.
  - Should not be too intrusive.

## Citation

If you found this library to be useful in academic work, then please cite:

```bibtex
@misc{pal2022lux,
    author = {Pal, Avik},
    title = {FluxMPI: Distributed Data Parallel Training of Neural Networks},
    year = {2021},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/avik-pal/FluxMPI.jl/}}
}
```

Also consider starring our github repo
