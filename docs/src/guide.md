# Guide to integrating FluxMPI into your code

There are essentially 6 main steps to remember:

1. Initialize FluxMPI [`FluxMPI.Init()`](@ref).

2. Sync Model Parameters and States [`FluxMPI.synchronize!(ps; root_rank)`](@ref).

3. Use [`DistributedDataContainer`](@ref) to distribute your data evenly across the
   processes. (Of course an alternative is to just manually partition your data.)

4. Wrap the optimizer in [`DistributedOptimizer`](@ref) or call 
   [`allreduce_gradients(gs::NamedTuple)`](@ref) before eveery `Optimisers.update`.

5. Sync the optimizer state across the processes
   [`FluxMPI.synchronize!(opt_state; root_rank)`](@ref).

6. Change logging code to check for [`local_rank`](@ref) == 0.

Finally, start the code using `mpiexecjl -n <np> julia --project=. <filename>.jl`
