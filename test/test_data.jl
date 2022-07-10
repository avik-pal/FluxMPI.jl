import FluxMPI, MPI, Random, Test

rng = Random.MersenneTwister()
Random.seed!(rng, 19)

FluxMPI.Init(; verbose=true)

Test.@testset "DistributedDataContainer" begin
  data = randn(rng, Float32, 10)
  tworkers = FluxMPI.total_workers()
  rank = FluxMPI.local_rank()

  dcontainer = FluxMPI.DistributedDataContainer(data)

  if rank != tworkers - 1
    Test.@test length(dcontainer) == ceil(length(data) / tworkers)
  else
    Test.@test length(dcontainer) ==
               length(data) - ceil(length(data) / tworkers) * (tworkers - 1)
  end

  dsum = 0
  for i in 1:length(dcontainer)
    dsum += dcontainer[i]
  end
  Test.@test isapprox(FluxMPI.MPIExtensions.allreduce!([dsum], +, MPI.COMM_WORLD)[1],
                      sum(data))
end

MPI.Finalize()
