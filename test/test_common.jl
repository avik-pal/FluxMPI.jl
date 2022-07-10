import FluxMPI, MPI, Test

FluxMPI.Init(; verbose=true)

Test.@testset "Common Utilities" begin
  Test.@test FluxMPI.Initialized()

  # Should always hold true
  Test.@test FluxMPI.local_rank() < FluxMPI.total_workers()

  Test.@test_nowarn FluxMPI.clean_println("Printing from Rank ", FluxMPI.local_rank())

  Test.@test_nowarn FluxMPI.clean_print("Printing from Rank ", FluxMPI.local_rank(), "\n")

  MPI.Finalize()
  Test.@test MPI.Finalized()
end
