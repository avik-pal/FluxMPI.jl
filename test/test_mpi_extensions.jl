import FluxMPI, MPI, Test

FluxMPI.Init(; verbose=true)

function _get_array_based_on_rank(dims; root_rank)
  if FluxMPI.local_rank() == root_rank
    return ones(dims...)
  else
    return zeros(dims...)
  end
end

Test.@testset "Iallreduce!" begin
  x = ones(4)

  y = similar(x)
  y, req = FluxMPI.MPIExtensions.Iallreduce!(x, y, +, MPI.COMM_WORLD)
  MPI.Wait!(req)

  Test.@test y == x .* FluxMPI.total_workers()

  y = similar(x)
  y, req = FluxMPI.MPIExtensions.Iallreduce!(x, y, *, MPI.COMM_WORLD)
  MPI.Wait!(req)

  Test.@test y == x
end

Test.@testset "Ibcast!" begin
  x = _get_array_based_on_rank((2, 3); root_rank=0)

  y, req = FluxMPI.MPIExtensions.Ibcast!(x, 0, MPI.COMM_WORLD)
  MPI.Wait!(req)

  Test.@test y == one.(x)
end

Test.@testset "Wrappers" begin
  Test.@testset "allreduce" begin
    x = ones(4)

    y = FluxMPI.MPIExtensions.allreduce!(copy(x), +, MPI.COMM_WORLD)
    Test.@test y == x .* FluxMPI.total_workers()

    y = FluxMPI.MPIExtensions.allreduce!(copy(x), *, MPI.COMM_WORLD)
    Test.@test y == x
  end

  Test.@testset "bcast" begin
    x = _get_array_based_on_rank((2, 3); root_rank=0)

    y = FluxMPI.MPIExtensions.bcast!(copy(x), 0, MPI.COMM_WORLD)
    Test.@test y == one.(x)
  end

  Test.@testset "reduce" begin
    x = ones(4)

    y = FluxMPI.MPIExtensions.reduce!(copy(x), +, 0, MPI.COMM_WORLD)
    if FluxMPI.local_rank() == 0
      Test.@test y == x .* FluxMPI.total_workers()
    else
      Test.@test y == x
    end
  end
end

MPI.Finalize()
