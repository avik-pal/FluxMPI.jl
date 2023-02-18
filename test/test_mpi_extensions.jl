using FluxMPI, MPI, Test

FluxMPI.Init(; verbose=true)

function _get_array_based_on_rank(dims; root_rank)
  return local_rank() == root_rank ? ones(dims...) : zeros(dims...)
end

@testset "Iallreduce!" begin
  x = ones(4)

  y = similar(x)
  y, req = FluxMPI.Iallreduce!(x, y, +, MPI.COMM_WORLD)
  MPI.Wait!(req)

  @test y == x .* total_workers()

  y = similar(x)
  y, req = FluxMPI.Iallreduce!(x, y, *, MPI.COMM_WORLD)
  MPI.Wait!(req)

  @test y == x
end

@testset "Ibcast!" begin
  x = _get_array_based_on_rank((2, 3); root_rank=0)

  y, req = FluxMPI.Ibcast!(x, 0, MPI.COMM_WORLD)
  MPI.Wait!(req)

  @test y == one.(x)
end

@testset "Wrappers" begin
  @testset "allreduce" begin
    x = ones(4)

    y = FluxMPI.allreduce!(copy(x), +, MPI.COMM_WORLD)
    @test y == x .* total_workers()

    y = FluxMPI.allreduce!(copy(x), *, MPI.COMM_WORLD)
    @test y == x
  end

  @testset "bcast" begin
    x = _get_array_based_on_rank((2, 3); root_rank=0)

    y = FluxMPI.bcast!(copy(x), 0, MPI.COMM_WORLD)
    @test y == one.(x)
  end

  @testset "reduce" begin
    x = ones(4)

    y = FluxMPI.reduce!(copy(x), +, 0, MPI.COMM_WORLD)
    if local_rank() == 0
      @test y == x .* total_workers()
    else
      @test y == x
    end
  end
end

MPI.Finalize()
