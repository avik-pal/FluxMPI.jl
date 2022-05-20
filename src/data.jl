"""
    DistributedDataContainer(data)

`data` must be compatible with `LearnBase` interface. The returned container is compatible with `LearnBase` interface and is used to partition the dataset across the available processes.
"""
struct DistributedDataContainer
    data::Any
    idxs::Any
end

function DistributedDataContainer(data)
    total_size = numobs(data)
    split_across = total_workers()
    size_per_process = Int(ceil(total_size / split_across))

    partitions = collect(Iterators.partition(1:total_size, size_per_process))
    idxs = collect(partitions[local_rank() + 1])

    return DistributedDataContainer(data, idxs)
end

Base.length(ddc::DistributedDataContainer) = length(ddc.idxs)

Base.getindex(ddc::DistributedDataContainer, i) = getobs(ddc.data, ddc.idxs[i])

# TODO: Needed to support DataLoaders.jl till (if?) it transitions to MLUtils
MLDataUtils.nobs(data::DistributedDataContainer) = numobs(data)

MLDataUtils.getobs(data::DistributedDataContainer, i, ::LearnBase.ObsDim.Undefined) = getobs(data, i)

MLDataUtils.getobs(data::DistributedDataContainer, i) = getobs(data, i)
