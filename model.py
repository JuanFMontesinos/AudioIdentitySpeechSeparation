from torchaudio.transforms import Spectrogram
import matplotlib.pyplot as plt
from einops import rearrange
from flerken.models import UNet
from torch import nn, istft
import torch

N_FFT = 1022
HOP_LENGTH = 256
AUDIO_LENGTH = 4 * 16384 - 1
AUDIO_LENGTH = 65526
K = 10


class Separator(nn.Module):
    def __init__(self):
        super().__init__()

        self.unet = UNet(input_channels=2,
                         output_channels=2,
                         layer_channels=[32, 64, 128, 256, 512],
                         architecture="sop",
                         film=(256, 'l'),
                         useBN=True,
                         mode="upsample")

        self.wav2sp = Spectrogram(n_fft=N_FFT, power=None, hop_length=HOP_LENGTH)
        self.register_buffer('_window', torch.hann_window(N_FFT))
        self.n_sources = 2

    def forward(self, inputs: dict):
        output = {'logits_mask': None,
                  'inference_mask': None,
                  'loss_mask': None,
                  'gt_mask': None,
                  'estimated_sp': None,
                  'estimated_wav': None}

        # Defining inputs
        srcm = inputs['audio']
        emb = inputs['embedding']

        # Computing fourier transform
        spm = self.wav2sp(srcm)  # Spectrogram main BxFxTx2

        # Creating artificial mixture
        B = spm.shape[0]  # Batch elements
        ndim = spm.ndim
        remix_coef = 0.5 if self.training else 1.0
        coef = (torch.rand(B, *[1 for _ in range(ndim - 1)], device=spm.device) < remix_coef).byte()
        sources = [spm, spm.flip(0) * coef]
        sp_mix_raw = sum(sources) / self.n_sources

        # Downsampling
        spm = spm[:, ::2, ...]
        sp_mix = sp_mix_raw[:, ::2, ...]  # BxFxTx2

        # Magnitude spectrogram for weighted loss
        mag = sp_mix.abs()  # Magnitude spectrogram BxFxT
        weight = torch.log1p(mag)
        weight = torch.clamp(weight, 1e-3, 10)

        x = rearrange(torch.view_as_real(sp_mix), 'b f t c -> b c f t')
        logits_mask = self.unet(x, emb)

        # Compute the loss mask and the ground truth mask
        gt_mask = self.complex_mask(spm, sp_mix)
        loss_mask = rearrange(self.tanh(logits_mask), 'b c f t  -> b f t c')
        output['loss'] = self.compute_loss(loss_mask, gt_mask, weight)

        output['logits_mask'] = logits_mask
        output['mix_sp'] = sp_mix_raw
        output['loss_mask'] = loss_mask
        output['gt_mask'] = gt_mask

        # Compute inference mask
        if not self.training:
            inference_mask = torch.view_as_complex(rearrange(logits_mask, 'b c f t  -> b f t c').contiguous())
            inference_mask = self.n_sources * inference_mask
            estimated_wav, estimated_sp = self.sp2wav(inference_mask.detach(), sp_mix_raw)
            raw_mix_wav = self.istft(sp_mix_raw)

            output['inference_mask'] = inference_mask
            output['estimated_sp'] = estimated_sp
            output['estimated_wav'] = estimated_wav
            output['raw_mix_wav'] = raw_mix_wav

        return output

    def istft(self, x):
        return istft(x, n_fft=N_FFT, hop_length=HOP_LENGTH, length=AUDIO_LENGTH, window=self._window)

    def sp2wav(self, inference_mask, mixture):
        if inference_mask.is_complex():
            inference_mask = torch.view_as_real(inference_mask).permute(0, 3, 1, 2)

        inference_mask = torch.nn.functional.upsample(inference_mask, scale_factor=(2, 1), mode='nearest').squeeze(1)
        inference_mask = torch.view_as_complex(inference_mask.permute(0, 2, 3, 1).contiguous())

        estimated_sp = inference_mask * mixture
        estimated_wav = self.istft(estimated_sp)
        return estimated_wav, estimated_sp

    def compute_loss(self, pred, gt, weight):
        assert pred.shape == gt.shape, 'Mask computation: Ground truth and predictions has to be the same shape'
        weight = weight.unsqueeze(-1)
        loss = (weight * (pred - gt).pow(2)).mean()
        return loss

    @torch.no_grad()
    def complex_mask(self, sp0, sp_mix, eps=torch.finfo(torch.float32).eps):
        # Bibliography about complex masks
        # http://homes.sice.indiana.edu/williads/publication_files/williamsonetal.cRM.2016.pdf
        assert sp0.shape == sp_mix.shape
        sp_mix += eps
        mask = torch.view_as_real(sp0 / sp_mix) / self.n_sources
        mask_bounded = self.tanh(mask)
        return mask_bounded

    @staticmethod
    def tanh(x):
        # *(1-torch.exp(-C * x))/(1+torch.exp(-C * x))
        # Compute this formula but using torch.tanh to deal with asymptotic values
        # Manually coded at https://github.com/vitrioil/Speech-Separation/blob/master/src/models/complex_mask_utils.py
        return K * torch.tanh(x)

    @torch.no_grad()
    def save_audio(self, batch_idx, waveform, path):
        assert waveform.ndim == 2
        from scipy.io.wavfile import write
        write(path, 16384, waveform[batch_idx].detach().cpu().numpy())

    @torch.no_grad()
    def save_loss_mask(self, batch_idx, loss_mask, gt_mask, path):
        gt_mask = torch.view_as_complex(gt_mask[batch_idx].detach().cpu())
        gt_mask_mag = gt_mask.abs().numpy()
        gt_mask_real = gt_mask.real.numpy()
        gt_mask_imag = gt_mask.imag.numpy()

        fig, ax = plt.subplots(nrows=2, ncols=3, constrained_layout=True)
        fig.set_size_inches(14, 8, forward=True)
        fig.subplots_adjust(right=0.8, )
        ax[0][0].set(title='Magnitude GT mask ')
        imm = ax[0][0].imshow(gt_mask_mag)
        ax[0][1].set(title='Real GT mask ')
        imr = ax[0][1].imshow(gt_mask_real)
        ax[0][2].set(title='Imag GT mask ')
        ax[0][2].imshow(gt_mask_imag)
        ax[0][2].label_outer()

        loss_mask = torch.view_as_complex(loss_mask[batch_idx].detach().cpu().contiguous())
        loss_mask_mag = loss_mask.abs().numpy()
        loss_mask_real = loss_mask.real.numpy()
        loss_mask_imag = loss_mask.imag.numpy()
        ax[1][0].set(title='Magnitude pred mask ')
        ax[1][0].imshow(loss_mask_mag)
        ax[1][1].set(title='Real pred mask ')
        ax[1][1].imshow(loss_mask_real)
        ax[1][2].set(title='Imag pred mask ')
        ax[1][2].imshow(loss_mask_imag)
        ax[1][2].label_outer()
        cbaxes = fig.add_axes([0.32, 0.12, 0.025, 0.75])
        cbaxes2 = fig.add_axes([0.64, 0.12, 0.025, 0.75])
        fig.colorbar(imm, ax=ax[:, 0], cax=cbaxes)
        fig.colorbar(imr, ax=ax[:, 1:], cax=cbaxes2)
        fig.tight_layout()
        fig.savefig(path, dpi=fig.dpi, bbox_inches='tight')
        plt.close('all')
