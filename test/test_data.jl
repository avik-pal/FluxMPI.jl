using FluxMPI, MPI, Random, Test

rng = Random.MersenneTwister()
Random.seed!(rng, 19)

FluxMPI.Init(; verbose=true)

@testset "DistributedDataContainer" begin
  data = randn(rng, Float32, 10)
  tworkers = total_workers()
  rank = local_rank()

  dcontainer = DistributedDataContainer(data)

  if rank != tworkers - 1
    @test length(dcontainer) == ceil(length(data) / tworkers)
  else
    @test length(dcontainer) ==
          length(data) - ceil(length(data) / tworkers) * (tworkers - 1)
  end

  dsum = 0
  for i in 1:length(dcontainer)
    dsum += dcontainer[i]
  end
  @test isapprox(FluxMPI.allreduce!([dsum], +, MPI.COMM_WORLD)[1], sum(data))
end

MPI.Finalize()
