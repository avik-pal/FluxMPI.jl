import FluxMPI, MPI, Optimisers, Test

FluxMPI.Init(; verbose=true)

Test.@testset "DistributedOptimizer" begin
  opt = Optimisers.Adam(0.001f0)
  ps = (a=zeros(4), b=zeros(4))
  st_opt = Optimisers.setup(opt, ps)

  dopt = FluxMPI.DistributedOptimizer(opt)
  st_dopt = Optimisers.setup(dopt, ps)

  Test.@test st_dopt.a.state == st_opt.a.state
  Test.@test st_dopt.b.state == st_opt.b.state

  Test.@test_nowarn FluxMPI.synchronize!(st_dopt; root_rank=0)

  gs = (a=ones(4), b=ones(4))

  _, ps_dopt = Optimisers.update(st_dopt, ps, gs)
  _, ps_opt = Optimisers.update(st_opt, ps,
                                (a=gs.a .* FluxMPI.total_workers(),
                                 b=gs.b .* FluxMPI.total_workers()))

  Test.@test isapprox(ps_dopt.a, ps_opt.a; atol=1.0e-5, rtol=1.0e-5)
  Test.@test isapprox(ps_dopt.b, ps_opt.b; atol=1.0e-5, rtol=1.0e-5)
end

Test.@testset "allreduce_gradients" begin
  gs = (a=ones(4), b=ones(4))

  gs_ = FluxMPI.allreduce_gradients(deepcopy(gs); on_gpu=false)

  Test.@test gs_.a == gs.a .* FluxMPI.total_workers()
  Test.@test gs_.b == gs.b .* FluxMPI.total_workers()
end

MPI.Finalize()
