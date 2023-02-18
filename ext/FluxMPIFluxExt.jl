module FluxMPIFluxExt

isdefined(Base, :get_extension) ? (using Flux) : (using ..Flux)
using FluxMPI, Functors

function FluxMPI.synchronize!(x::FluxMPIFluxModel; root_rank::Integer=0)
  return fmap(x -> FluxMPI.synchronize!(x; root_rank), x.model)
end

end
