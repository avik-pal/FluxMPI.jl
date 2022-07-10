import ComponentArrays, FluxMPI, MPI, Optimisers, Test

FluxMPI.Init(; verbose=true)

function _get_array_based_on_rank(dims; root_rank)
  if FluxMPI.local_rank() == root_rank
    return ones(dims...)
  else
    return zeros(dims...)
  end
end

Test.@testset "synchronize" begin
  # synchronize!(x::AbstractArray; root_rank::Integer=0)
  # synchronize!(x::ComponentArray; root_rank::Integer=0)
  root_rank = 0

  Test.@testset "NamedTuple" begin
    gs = (a=(b=_get_array_based_on_rank((2, 3); root_rank),
             c=_get_array_based_on_rank((2, 3); root_rank)),
          d=_get_array_based_on_rank((2, 3); root_rank))

    gs_ = FluxMPI.synchronize!(gs; root_rank)

    Test.@test all(gs_.a.b .== 1)
    Test.@test all(gs_.a.c .== 1)
    Test.@test all(gs_.d .== 1)

    Test.@testset "Optimisers" begin
      opt = Optimisers.Adam(0.001f0)
      st_opt = Optimisers.setup(opt, gs)

      if FluxMPI.local_rank() == root_rank
        st_opt.a.b.state[1] .= 1
        st_opt.a.b.state[2] .= 1
        st_opt.a.c.state[1] .= 1
        st_opt.a.c.state[2] .= 1
        st_opt.d.state[1] .= 1
        st_opt.d.state[2] .= 1
      end

      st_opt = FluxMPI.synchronize!(st_opt; root_rank)

      Test.@test all(st_opt.a.b.state[1] .== 1)
      Test.@test all(st_opt.a.b.state[2] .== 1)
      Test.@test all(st_opt.a.c.state[1] .== 1)
      Test.@test all(st_opt.a.c.state[2] .== 1)
      Test.@test all(st_opt.d.state[1] .== 1)
      Test.@test all(st_opt.d.state[2] .== 1)

      # Has no state
      opt = Optimisers.Descent(0.001f0)
      st_opt = Optimisers.setup(opt, gs)

      Test.@test_nowarn FluxMPI.synchronize!(st_opt; root_rank)
    end

    Test.@testset "ComponentArray" begin
      gs = (a=(b=_get_array_based_on_rank((2, 3); root_rank),
               c=_get_array_based_on_rank((2, 3); root_rank)),
            d=_get_array_based_on_rank((2, 3); root_rank))
      cgs = ComponentArrays.ComponentArray(gs)
      cgs_ = FluxMPI.synchronize!(cgs; root_rank)

      Test.@test all(cgs_.a.b .== 1)
      Test.@test all(cgs_.a.c .== 1)
      Test.@test all(cgs_.d .== 1)
    end
  end

  Test.@testset "Tuple" begin
    gs = ((_get_array_based_on_rank((2, 3); root_rank),
           _get_array_based_on_rank((2, 3); root_rank)),
          _get_array_based_on_rank((2, 3); root_rank))

    gs = FluxMPI.synchronize!(gs; root_rank)

    Test.@test all(gs[1][1] .== 1)
    Test.@test all(gs[1][2] .== 1)
    Test.@test all(gs[2] .== 1)
  end

  Test.@testset "Misc." begin
    x = nothing
    x = FluxMPI.synchronize!(x; root_rank)
    Test.@test x == nothing

    if root_rank == FluxMPI.local_rank()
      x = :x
    else
      x = :y
    end
    x_ = FluxMPI.synchronize!(x; root_rank)
    # Symbol should not change
    Test.@test x_ == x

    x = FluxMPI.synchronize!(FluxMPI.local_rank(); root_rank)
    Test.@test x == root_rank
  end
end

MPI.Finalize()
