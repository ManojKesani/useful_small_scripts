from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

real_batch = next(iter(dataloader))

writer.add_graph(netD, real_batch[0].to(device))
writer.close()

%load_ext tensorboard
%tensorboard --logdir=runs
