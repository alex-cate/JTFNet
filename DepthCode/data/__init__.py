from importlib import import_module

# from dataloader import MSDataLoader
# from torch.utils.data.dataloader import default_collate
from torch.utils.data import dataloader


class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            trainset = getattr(module_train, args.data_train)(args)
            self.loader_train = dataloader.DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )
        if args.data_test in ['art', 'books', 'moebius', 'dolls', 'laundry', 'reindeer', 'tsukuba', 'venus', 'teddy', 'cones']:
            if not args.benchmark_noise:
                module_test = import_module('data.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, name=args.data_test, train=False)
            else:
                module_test = import_module('data.benchmark_noise')
                testset = getattr(module_test, 'BenchmarkNoise')(
                    args,
                    train=False
                )

        else:
            # module_test = import_module('data.' + d.lower())
            # testset = getattr(module_test, d)(args, train=False)
            module_test = import_module('data.testdata')
            testset = getattr(module_test, 'Testdata')(args, name=args.data_test, train=False)
        self.loader_test = dataloader.DataLoader(
            testset,
            batch_size=1,
            pin_memory=not args.cpu,
            num_workers=args.n_threads,
        )
