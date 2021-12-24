import os
import argparse

import torch
from torch_mir_eval import bss_eval_sources
from dataset import dataloader
from model import Separator
from flerken.framework.framework import Experiment, Trainer


class PlaceholderLoss(torch.nn.Module):
    def forward(self, pred, gt):
        sep_loss = pred['loss']
        return sep_loss


class InhTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(InhTrainer, self).__init__(*args, **kwargs)
        self.criterion = PlaceholderLoss()
        self.EPOCHS = 1000
        self._model.initializable_layers = []

    def loader_mapping(self, x):
        return {'gt': None, 'inputs': [x], 'vs': None}

    def backpropagate(self, debug):
        self.optimizer.zero_grad()
        if debug['isnan']:
            assert not torch.isnan(self.loss).any()
            assert not torch.isinf(self.loss).any()
        if not torch.isnan(self.loss).any() and not torch.isinf(self.loss).any():
            self.loss.backward()
            self.optimizer.step()

    def hook(self, vrs) -> None:
        path = os.path.join(self.IO.workdir, 'dump_files')
        waveforms = vrs['pred']['estimated_wav']
        ground_truth = vrs['inputs'][0]['audio']
        mixture = vrs['pred']['raw_mix_wav']

        sdr_pred, sir_pred, sar_pred, _ = bss_eval_sources(ground_truth.unsqueeze(1), waveforms.unsqueeze(1),
                                                           compute_permutation=False)
        sdr_mix, sir_mix, sar_mix, _ = bss_eval_sources(ground_truth.unsqueeze(1), mixture.unsqueeze(1),
                                                        compute_permutation=False)
        print((sdr_pred - sdr_mix).mean())

        for i in range(waveforms.shape[0]):
            if not os.path.exists(os.path.join(path, str(i))):
                os.makedirs(os.path.join(path, str(i)))
            self.model.save_audio(i, waveforms, os.path.join(path, str(i), 'estimated_wav.wav'))
            self.model.save_audio(i, ground_truth, os.path.join(path, str(i), 'ground_truth.wav'))
            self.model.save_audio(i, mixture, os.path.join(path, str(i), 'mixture.wav'))


def dataset_path(suffix):
    return f'/media/jfm/SlaveEVO970/acappella/splits/{suffix}/audio'


def argparse_default():
    parser = argparse.ArgumentParser(description='U-Net training')

    parser.add_argument('--workname', help='Experiment name', type=str, default=None)
    parser.add_argument('--arxiv_path', help='Main directory for all the experiments',
                        type=str, default='./debug_dir')
    parser.add_argument('--pretrained_from', help='Use some weights to start from',
                        type=str, default=None)
    parser.add_argument('--device', help='Device for training the experiments',
                        type=str, default='cuda:0')
    parser.add_argument('--testing', dest='testing', action='store_true')
    parser.set_defaults(testing=False)
    args = parser.parse_args()
    return args


# Define default values
args = argparse_default()

# Set GPU device
device = torch.device(args.device)

# Create experiment manager
ex = Experiment(args.arxiv_path, args.workname)
if ex.resume_from is None:
    ex.IO.add_cfg('argparse', args.__dict__)

# Instantiating the model
model = Separator()

# Instantiating the dataloader
train_dataloader = dataloader(dataset_path('train'), batch_size=100, debug=True, duplicate=True)
test_dataloader = dataloader(dataset_path('test_unseen'), batch_size=100, debug=True)

# Setting the trainer
trainer = InhTrainer(device, model)
trainer.dump_files = args.testing
if args.pretrained_from is not None:
    trainer._model.load(args.pretrained_from, strict_loading=True, from_checkpoint=True)
trainer.model.to(device)

# Defining the optimizer
trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

with ex.autoconfig(trainer) as trainer:
    if args.testing:
        trainer.epoch = 0
        with torch.no_grad():
            trainer.run_epoch(test_dataloader, 'test',
                              backprop=False,
                              metrics=['loss'],
                              checkpoint=trainer.checkpoint(metric='loss', freq=20))
        quit()
    for trainer.epoch in range(trainer.start_epoch, trainer.EPOCHS):
        trainer.run_epoch(train_dataloader, 'train',
                          backprop=True,
                          metrics=['loss'])
        with torch.no_grad():
            trainer.run_epoch(test_dataloader, 'test',
                              backprop=False,
                              metrics=['loss'],
                              checkpoint=trainer.checkpoint(metric='loss', freq=20))
