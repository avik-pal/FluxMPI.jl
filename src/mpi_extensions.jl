# NOTE(@avik-pal): Data Movement between CPU and GPU. `Lux` & `Flux` have much better
#                  support for this but this is all we need here. If not an abstract array
#                  don't do anything.
# Maybe we want to have a central repository containing these device transfer utilities
cpu(x) = x
gpu(x) = x
cpu(x::AbstractArray) = adapt(Array, x)
gpu(x::AbstractArray) = adapt(CuArray, x)

## Certain Non-blocking collective communication primitives
"""
    Iallreduce!(sendbuf, recvbuf, op, comm)
    Iallreduce!(sendrecvbuf, op, comm)

Performs non-blocking elementwise reduction using the operator `op` on the buffer `sendbuf`.
`sendbuf` can also be a scalar, in which case `recvbuf` will be a value of the same type.

`recvbuf` and an MPI_Request object are returned. The value in `recvbuf` is only valid after
the request has been completed. (`MPI.Wait!`)

!!! warning

    OpenMPI doesn't support Iallreduce! with CUDA. See
    [this issue](https://github.com/open-mpi/ompi/issues/9845).
"""
function Iallreduce!(rbuf::MPI.RBuffer, op::Union{MPI.Op, MPI.MPI_Op}, comm::MPI.Comm)
  req = MPI.Request()
  # int MPI_Iallreduce(const void *sendbuf, void *recvbuf, int count,
  #                    MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
  #                    MPI_Request * request)
  MPI.@mpichk ccall((:MPI_Iallreduce, MPI.libmpi), Cint,
                    (MPI.MPIPtr, MPI.MPIPtr, Cint, MPI.MPI_Datatype, MPI.MPI_Op,
                     MPI.MPI_Comm, Ptr{MPI.MPI_Request}), rbuf.senddata, rbuf.recvdata,
                    rbuf.count, rbuf.datatype, op, comm, req)
  req.buffer = rbuf
  finalizer(MPI.free, req)
  return rbuf.recvdata, req
end

function Iallreduce!(rbuf::MPI.RBuffer, op, comm::MPI.Comm)
  return Iallreduce!(rbuf, MPI.Op(op, eltype(rbuf)), comm)
end

function Iallreduce!(sendbuf, recvbuf, op, comm::MPI.Comm)
  return Iallreduce!(MPI.RBuffer(sendbuf, recvbuf), op, comm)
end

Iallreduce!(buf, op, comm::MPI.Comm) = Iallreduce!(MPI.IN_PLACE, buf, op, comm)

"""
    Ibcast!(buf, root, comm)

Non-blocking broadcast of the buffer `buf` to all processes in `comm`.

`recvbuf` and an MPI_Request object are returned. The value in `recvbuf` is only valid after
the request has been completed. (`MPI.Wait!`)
"""
function Ibcast!(buf::MPI.Buffer, root::Integer, comm::MPI.Comm)
  req = MPI.Request()
  # int MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root,
  #                MPI_Comm comm, MPI_Request *request)
  MPI.@mpichk ccall((:MPI_Ibcast, MPI.libmpi), Cint,
                    (MPI.MPIPtr, Cint, MPI.MPI_Datatype, Cint, MPI.MPI_Comm,
                     Ptr{MPI.MPI_Request}), buf.data, buf.count, buf.datatype, root, comm,
                    req)
  req.buffer = buf
  finalizer(MPI.free, req)
  return buf.data, req
end

Ibcast!(buf, root::Integer, comm::MPI.Comm) = Ibcast!(MPI.Buffer(buf), root, comm)

# Simpler wrappers
"""
    allreduce!(v, op, comm)

Perform `MPI.Allreduce!` ensuring that CuArrays are safely transfered to CPU if CUDA-aware
MPI is unavailable/disabled.
"""
function allreduce!(v::CuArray, op::Any, comm::MPI.Comm)
  @static if !MPI_IS_CUDA_AWARE[]
    v = cpu(v)
  end
  MPI.Allreduce!(v, op, comm)
  @static if !MPI_IS_CUDA_AWARE[]
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

Perform `MPI.Bcast!` ensuring that CuArrays are safely transfered to CPU if CUDA-aware MPI
is unavailable/disabled.
"""
function bcast!(v::CuArray, root::Integer, comm::MPI.Comm)
  @static if !MPI_IS_CUDA_AWARE[]
    v = cpu(v)
  end
  MPI.Bcast!(v, root, comm)
  @static if !MPI_IS_CUDA_AWARE[]
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

Perform `MPI.Reduce!` ensuring that CuArrays are safely transfered to CPU if CUDA-aware MPI
is unavailable/disabled.
"""
function reduce!(v::CuArray, op::Any, root::Integer, comm::MPI.Comm)
  @static if !MPI_IS_CUDA_AWARE[]
    v = cpu(v)
  end
  MPI.Reduce!(v, op, root, comm)
  @static if !MPI_IS_CUDA_AWARE[]
    v = gpu(v)
  end
  return v
end

function reduce!(v::AbstractArray, op::Any, root::Integer, comm::MPI.Comm)
  MPI.Reduce!(v, op, root, comm)
  return v
end
