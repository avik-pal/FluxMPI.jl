using FluxMPI, MPI, Test

FluxMPI.Init(; verbose=true)

@testset "Common Utilities" begin
  @test FluxMPI.Initialized()

  # Should always hold true
  @test FluxMPI.local_rank() < FluxMPI.total_workers()

  @test_nowarn FluxMPI.fluxmpi_println("Printing from Rank ", FluxMPI.local_rank())

  @test_nowarn FluxMPI.fluxmpi_print("Printing from Rank ", FluxMPI.local_rank(), "\n")

  MPI.Finalize()
  @test MPI.Finalized()
end
