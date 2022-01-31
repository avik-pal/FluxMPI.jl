using FastAI, FluxMPI

FluxMPI.Init()

data, blocks = loaddataset("imagenette2-160", (Image, Label))
data_ddp = DistributedDataContainer(data)

method = ImageClassificationSingle(blocks)
learner = methodlearner(method, data_ddp; callbacks=[ToGPU(), Metrics(accuracy)],
                        optimizer=DistributedOptimiser(ADAM()))

fitonecycle!(learner, 10)
