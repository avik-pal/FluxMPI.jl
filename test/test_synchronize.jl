using ComponentArrays, FluxMPI, MPI, Optimisers, Test

FluxMPI.Init(; verbose=true)

function _get_array_based_on_rank(dims; root_rank)
  if FluxMPI.local_rank() == root_rank
    return ones(dims...)
  else
    return zeros(dims...)
  end
end

@testset "synchronize" begin
  root_rank = 0

  @testset "NamedTuple" begin
    gs = (a=(b=_get_array_based_on_rank((2, 3); root_rank),
             c=_get_array_based_on_rank((2, 3); root_rank)),
          d=_get_array_based_on_rank((2, 3); root_rank))

    gs_ = FluxMPI.synchronize!(gs; root_rank)

    @test all(gs_.a.b .== 1)
    @test all(gs_.a.c .== 1)
    @test all(gs_.d .== 1)

    @testset "Optimisers" begin
      opt = Adam(0.001f0)
      st_opt = Optimisers.setup(opt, gs)

      if local_rank() == root_rank
        st_opt.a.b.state[1] .= 1
        st_opt.a.b.state[2] .= 1
        st_opt.a.c.state[1] .= 1
        st_opt.a.c.state[2] .= 1
        st_opt.d.state[1] .= 1
        st_opt.d.state[2] .= 1
      end

      st_opt = FluxMPI.synchronize!(st_opt; root_rank)

      @test all(st_opt.a.b.state[1] .== 1)
      @test all(st_opt.a.b.state[2] .== 1)
      @test all(st_opt.a.c.state[1] .== 1)
      @test all(st_opt.a.c.state[2] .== 1)
      @test all(st_opt.d.state[1] .== 1)
      @test all(st_opt.d.state[2] .== 1)

      # Has no state
      opt = Descent(0.001f0)
      st_opt = Optimisers.setup(opt, gs)

      @test_nowarn FluxMPI.synchronize!(st_opt; root_rank)
    end

    @testset "ComponentArray" begin
      gs = (a=(b=_get_array_based_on_rank((2, 3); root_rank),
               c=_get_array_based_on_rank((2, 3); root_rank)),
            d=_get_array_based_on_rank((2, 3); root_rank))
      cgs = ComponentArray(gs)
      cgs_ = FluxMPI.synchronize!(cgs; root_rank)

      @test all(cgs_.a.b .== 1)
      @test all(cgs_.a.c .== 1)
      @test all(cgs_.d .== 1)
    end
  end

  @testset "Tuple" begin
    gs = ((_get_array_based_on_rank((2, 3); root_rank),
           _get_array_based_on_rank((2, 3); root_rank)),
          _get_array_based_on_rank((2, 3); root_rank))

    gs = FluxMPI.synchronize!(gs; root_rank)

    @test all(gs[1][1] .== 1)
    @test all(gs[1][2] .== 1)
    @test all(gs[2] .== 1)
  end

  @testset "Misc." begin
    x = nothing
    x = FluxMPI.synchronize!(x; root_rank)
    @test x == nothing

    if root_rank == local_rank()
      x = :x
    else
      x = :y
    end
    x_ = FluxMPI.synchronize!(x; root_rank)
    # Symbol should not change
    @test x_ == x

    x = FluxMPI.synchronize!(local_rank(); root_rank)
    @test x == root_rank
  end
end

MPI.Finalize()
