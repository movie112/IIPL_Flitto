#!/usr/bin/env python3
import os
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from models.model.diarization import Diarization
from utils.feature_extraction import extract_fbank
from trainer.utils.rttm_load import rttm_load
from utils.ckpt_load import ckpt_load
import hyperpyyaml
import warnings

warnings.filterwarnings('ignore')

def main():
    parser = ArgumentParser()
    parser.add_argument('--OMP_NUM_THREADS', type=int, default=1)
    parser.add_argument('--wav_scp', type=str,
                        default='.../KR_wav.scp',
                        help="wav.scp 파일 경로 ('utt_id wav_path')")
    parser.add_argument('--configs', type=str,
                        default='.../SD_infer.yaml',
                        help='YAML configuration file path')
    parser.add_argument('--test_from_folder', type=str,
                        default='.../logs/KR',
                        help='Checkpoint directory for averaging')
    parser.add_argument('--output_rttm', type=str,
                        default='.../KR.rttm',
                        help='최종 RTTM 출력 파일 경로')
    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = str(args.OMP_NUM_THREADS)

    with open(args.configs, 'r') as f:
        configs = hyperpyyaml.load_hyperpyyaml(f)

    print('Averaging checkpoints and loading model...')
    state_dict = ckpt_load(test_folder=args.test_from_folder)
    stripped = {
        (k[len('model.'):]): v if k.startswith('model.') else v
        for k, v in state_dict.items()
    }
    model = Diarization(
        n_speakers=configs['data']['num_speakers'],
        in_size=(2 * configs['data']['context_recp'] + 1) * configs['data']['feat']['n_mels'],
        **configs['model']['params']
    )
    model.load_state_dict(stripped)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with open(args.wav_scp, 'r') as f:
        entries = [line.strip().split() for line in f if len(line.split()) == 2]

    with open(args.output_rttm, 'w', encoding='utf-8') as fout:
        for utt_id, wav_path in tqdm(entries, desc='Diarizing'):
            try:
                feat = extract_fbank(
                    wav_path,
                    context_size=configs['data']['context_recp'],
                    input_transform=configs['data']['feat_type'],
                    frame_size=configs['data']['feat']['win_length'],
                    frame_shift=configs['data']['feat']['hop_length'],
                    subsampling=configs['data']['subsampling']
                )
                if isinstance(feat, torch.Tensor):
                    feat_np = feat.cpu().numpy().astype('float32')
                else:
                    feat_np = feat.astype('float32')

                feat_tensor = torch.from_numpy(feat_np).to(device)

                clip_len = feat_tensor.shape[0]

                with torch.no_grad():
                    preds_list, _, _ = model.test(
                        [feat_tensor],
                        [clip_len],
                        configs['data']['max_speakers'] + 2
                    )

                raw = preds_list[0]
                if isinstance(raw, torch.Tensor):
                    raw = raw.cpu().numpy()
                prob = torch.from_numpy(raw[:, 1:]).float()
                pred = torch.sigmoid(prob)

                rttm_dict = rttm_load(
                    rec=utt_id,
                    pred=pred,
                    frame_shift=configs['data']['feat']['hop_length'],
                    subsampling=configs['data']['subsampling'],
                    sampling_rate=configs['data']['feat']['sample_rate']
                )
                lines = rttm_dict.get(utt_id, [])
                for line in lines:
                    fout.write(line + '\n')
                fout.flush()

            except Exception as e:
                print(f"[Warning] {utt_id} skipped: {e}")

    print(f"Done. RTTM saved to: {args.output_rttm}")


if __name__ == '__main__':
    main()
