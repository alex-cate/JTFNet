import torch
from thop import profile
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    print('Total params: %2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # input_color = torch.randn(8, 3, 128, 128)
    # input_depth = torch.randn(8, 1, 32, 32)
    # flop, para = profile(model, (input_depth,input_color,0,))
    # print('Flop:%2fG Params: %2fM' % ((flop / 1000000000.0),(para / 1000000.0)))
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()
        t.scheduler.step()

    checkpoint.done()
