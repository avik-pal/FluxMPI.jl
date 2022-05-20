module MPIExtensions

using Adapt: Adapt
using CUDA: CUDA
using MPI: MPI

const mpi_is_cuda_aware = Ref(false)

# Data Movement between CPU and GPU. `Lux` & `Flux` have much better support for this but this is all we need here
cpu(x::AbstractArray) = Adapt.adapt(Array, x)
gpu(x::AbstractArray) = Adapt.adapt(CUDA.CuArray, x)

function __init__()
    # If using `mpi_is_cuda_aware` anywhere use the @static macro since we anyways need to recompile the code when MPI implementation changes
    disable_cuda_aware_support = parse(Bool, get(ENV, "FLUXMPI_DISABLE_CUDAMPI_SUPPORT", "false"))
    mpi_is_cuda_aware[] = !disable_cuda_aware_support && MPI.has_cuda()
    if disable_cuda_aware_support && MPI.has_cuda()
        @info "CUDA-aware MPI support disabled using `FLUXMPI_DISABLE_CUDAMPI_SUPPORT=true`" maxlog = 1
    elseif !mpi_is_cuda_aware[]
        @warn "MPI Implementation is not CUDA Aware" maxlog = 1
    end
end

# Certain Non-blocking collective communication primitives
# TODO: Should be moved to MPI.jl?
"""
    Iallreduce!(sendbuf, recvbuf, op, comm)
    Iallreduce!(sendrecvbuf, op, comm)

Performs non-blocking elementwise reduction using the operator `op` on the buffer `sendbuf`. `sendbuf` can also be a scalar, in which case `recvbuf` will be a value of the same type.

`recvbuf` and an MPI_Request object are returned. The value in `recvbuf` is only valid after the request has been completed. (`MPI.Wait!`)

!!! warning
    OpenMPI doesn't support Iallreduce! with CUDA. See https://github.com/open-mpi/ompi/issues/9845
"""
function Iallreduce!(rbuf::MPI.RBuffer, op::Union{MPI.Op,MPI.MPI_Op}, comm::MPI.Comm)
    req = MPI.Request()
    # int MPI_Iallreduce(const void *sendbuf, void *recvbuf, int count,
    #                    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
    #                    MPI_Request * request)
    MPI.@mpichk ccall(
        (:MPI_Iallreduce, MPI.libmpi),
        Cint,
        (MPI.MPIPtr, MPI.MPIPtr, Cint, MPI.MPI_Datatype, MPI.MPI_Op, MPI.MPI_Comm, Ptr{MPI.MPI_Request}),
        rbuf.senddata,
        rbuf.recvdata,
        rbuf.count,
        rbuf.datatype,
        op,
        comm,
        req,
    )
    req.buffer = rbuf
    finalizer(free, req)
    return rbuf.recvdata, req
end

Iallreduce!(rbuf::MPI.RBuffer, op, comm::MPI.Comm) = Iallreduce!(rbuf, MPI.Op(op, eltype(rbuf)), comm)

Iallreduce!(sendbuf, recvbuf, op, comm::MPI.Comm) = Iallreduce!(MPI.RBuffer(sendbuf, recvbuf), op, comm)

Iallreduce!(buf, op, comm::MPI.Comm) = Iallreduce!(MPI.IN_PLACE, buf, op, comm)

"""
    Ibcast!(buf, root, comm)

Non-blocking broadcast of the buffer `buf` to all processes in `comm`.

`recvbuf` and an MPI_Request object are returned. The value in `recvbuf` is only valid after the request has been completed. (`MPI.Wait!`)
"""
function Ibcast!(buf::MPI.Buffer, root::Integer, comm::MPI.Comm)
    req = MPI.Request()
    # int MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root,
    #                MPI_Comm comm, MPI_Request *request)
    MPI.@mpichk ccall(
        (:MPI_Ibcast, MPI.libmpi),
        Cint,
        (MPI.MPIPtr, Cint, MPI.MPI_Datatype, Cint, MPI.MPI_Comm, Ptr{MPI.MPI_Request}),
        buf.data,
        buf.count,
        buf.datatype,
        root,
        comm,
        req,
    )
    req.buffer = buf
    finalizer(free, req)
    return buf.data, req
end

Ibcast!(buf, root::Integer, comm::MPI.Comm) = Ibcast!(MPI.Buffer(buf), root, comm)

"""
    allreduce!(v, op, comm)

Perform `MPI.Allreduce!` ensuring that CuArrays are safely transfered to CPU if CUDA-aware MPI is unavailable/disabled.
"""
function allreduce!(v::CUDA.CuArray, op::Any, comm::MPI.Comm)
    @static if !mpi_is_cuda_aware[]
        v = cpu(v)
    end
    MPI.Allreduce!(v, op, comm)
    @static if !mpi_is_cuda_aware[]
        v = gpu(v)
    end
    return v
end

function allreduce!(v::AbstractArray, op::Any, comm::MPI.Comm)
    MPI.Allreduce!(v, op, comm)
    return v
end

"""
    bcast!(v, op, comm)

Perform `MPI.Bcast!` ensuring that CuArrays are safely transfered to CPU if CUDA-aware MPI is unavailable/disabled.
"""
function bcast!(v::CUDA.CuArray, root::Integer, comm::MPI.Comm)
    @static if !mpi_is_cuda_aware[]
        v = cpu(v)
    end
    MPI.Bcast!(v, root, comm)
    @static if !mpi_is_cuda_aware[]
        v = gpu(v)
    end
    return v
end

function bcast!(v::AbstractArray, root::Integer, comm::MPI.Comm)
    MPI.Bcast!(v, root, comm)
    return v
end

"""
    reduce!(v, op, comm)

Perform `MPI.Reduce!` ensuring that CuArrays are safely transfered to CPU if CUDA-aware MPI is unavailable/disabled.
"""
function reduce!(v::CUDA.CuArray, op::Any, root::Integer, comm::MPI.Comm)
    @static if !mpi_is_cuda_aware[]
        v = cpu(v)
    end
    MPI.Reduce!(v, op, root, comm)
    @static if !mpi_is_cuda_aware[]
        v = gpu(v)
    end
    return v
end

function reduce!(v::AbstractArray, op::Any, root::Integer, comm::MPI.Comm)
    MPI.Reduce!(v, op, root, comm)
    return v
end

end
