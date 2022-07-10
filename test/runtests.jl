import FluxMPI, MPI, Test

nprocs_str = get(ENV, "JULIA_MPI_TEST_NPROCS", "")
nprocs = nprocs_str == "" ? clamp(Sys.CPU_THREADS, 2, 4) : parse(Int, nprocs_str)
testdir = @__DIR__
istest(f) = endswith(f, ".jl") && startswith(f, "test_")
testfiles = sort(filter(istest, readdir(testdir)))

@info "Running FluxMPI tests" nprocs

Test.@testset "$f" for f in testfiles
  MPI.mpiexec() do cmd
    run(`$cmd -n $nprocs $(Base.julia_cmd()) $(joinpath(testdir, f))`)
    Test.@test true
  end
end
