from scipy.signal import medfilt
import torch
import torch.nn.functional as F

def rttm_load(rec, pred, frame_shift=80, threshold=0.5, median=11, subsampling=10, sampling_rate=8000):
    pred = torch.where(pred > threshold, 1, 0)
    if median > 1:
        pred = medfilt(pred, (median, 1))
        pred = torch.from_numpy(pred).float()

    fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
    events = []

    for spkid, frames in enumerate(pred.T):
        if spkid == pred.shape[1] - 1:
            continue
        frames = F.pad(frames, (1, 1), 'constant')
        changes, = torch.where(torch.diff(frames, dim=0) != 0)
        for s, e in zip(changes[::2], changes[1::2]):
            start = s * frame_shift * subsampling / sampling_rate
            duration = (e - s) * frame_shift * subsampling / sampling_rate
            line = fmt.format(rec, start, duration, rec + "_" + str(spkid))
            events.append((start, line))

    events.sort(key=lambda x: x[0])
    rttm = {rec: [line for _, line in events]}
    return rttm