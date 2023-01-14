import os
import math

from decimal import Decimal
import time
import utility
import imageio
import torch
from torch.autograd import Variable
from tqdm import tqdm


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp  # checkpoint
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        # print(self.args.load)
        if self.args.load != '.':
            # print("loading")
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (depth_hr, depth_lr, color, idx_scale) in enumerate(self.loader_train):
            depth_hr, depth_lr, color = self.prepare([depth_hr, depth_lr, color])
            # depth_hr, depth_lr = self.prepare([depth_hr, depth_lr])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            # depth_sr = self.model(depth_lr, color, idx_scale)
            depth_sr,_,_ = self.model(depth_lr, color, idx_scale)
            # depth_sr = self.model(depth_lr, idx_scale)
            # patch = utility.quantize(depth_sr, self.args.rgb_range)
            # patch_np= patch[0].byte().permute(1, 2, 0).cpu().numpy()
            # imageio.imsave('/DISK/wh/Depth/train_img/{}_{}.png'.format(epoch,batch),patch_np)
            loss = self.loss(depth_sr, depth_hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()
        time_sum = 0
        timer_test = utility.timer()
        # if self.args.save_results: self.ckp.begin_background()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                # d.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (depth_hr, depth_lr, color, filename) in enumerate(tqdm_test):
                    # for depth_lr, depth_hr, color, filename in tqdm(d, ncols=80):
                    depth_hr, depth_lr, color = self.prepare([depth_hr, depth_lr, color])
                    # depth_hr, depth_lr = self.prepare([depth_hr, depth_lr])

                    # patch = utility.quantize(depth_lr, self.args.rgb_range)
                    # patch_np = patch[0].byte().permute(1, 2, 0).cpu().numpy()
                    # imageio.imsave('/DISK/wh/Depth/test_img/{}.png'.format(filename[0]), patch_np)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    # depth_sr = self.model(depth_lr, color, idx_scale)
                    depth_sr,_,_ = self.model(depth_lr, color, idx_scale)
                    # depth_sr = self.model(depth_lr, idx_scale)
                    torch.cuda.synchronize()
                    t2 = time.time()
                    t = t2 - t1
                    time_sum = time_sum + t
                    depth_sr = utility.quantize(depth_sr, self.args.rgb_range)
                    depth_hr = utility.quantize(depth_hr, self.args.rgb_range)

                    save_list = [depth_sr]
                    eval_acc += utility.calc_rmse(depth_sr, depth_hr)
                    if self.args.save_results:
                        self.ckp.save_results(filename[0], save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.min(0)
                self.ckp.write_log(
                    '[{} x{}]\tRMSE: {:.4f} (Best: {:.4f} @epoch {})'.format(
                        filename[0],
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        print(time_sum)
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

        # if self.args.save_results:
        #     self.ckp.end_background()

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
