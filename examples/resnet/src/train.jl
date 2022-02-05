using Flux, FluxMPI, CUDA, Metalhead

FluxMPI.Init()

model = ResNet50() |> gpu
ps = Flux.params(model)
broadcast_parameters(model; root_rank = 0)

batchsize = 8
data = [(rand(Float32, 224, 224, 3, batchsize) |> gpu, onehotbatch(rand(1:1000), 1:1000)) |> gpu for _ in 1:3]

opt = DistributedOptimiser(ADAM())

loss(x, y, m) = Flux.Losses.logitcrossentropy(m(x), y)

for (i, (x, y)) in enumerate(data)
    @info "Starting batch $i ..."
    gs = gradient(() -> loss(x, y, model), ps)
    Flux.update!(opt, ps, gs)
end
