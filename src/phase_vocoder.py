from copy import copy
import math
import numpy as np
from sys import float_info


def phase_vocoder(x, stretch_factor, frame_size=1024):
    """Time-stretch a signal x by a given stretch factor."""

    EPS = float_info.epsilon

    hop_size = math.ceil(frame_size / 4)
    hN = math.floor(frame_size / 2) + 1
    window = np.hanning(frame_size) + [0]

    # Analysis hops faster or slower than synthesis, to time- squish or stretch.
    analysis_hop_size = math.ceil(hop_size / stretch_factor)

    num_frames = math.ceil(len(x) / analysis_hop_size)

    y_num_samples = (num_frames - 1) * hop_size + frame_size
    y = np.zeros(y_num_samples)

    tmp = (num_frames - 1) * analysis_hop_size + frame_size
    samples_to_pad = tmp - len(x)
    samples_to_pad = np.int(samples_to_pad)

    # Pad beginning of `x` to facilitate first pivot frame.
    x = np.pad(x, [hop_size, samples_to_pad])

    # Initialize variables.
    Y_last = np.zeros(hN)

    # Setting up pointers.
    pivot_in = 0
    pivot_out = pivot_in + frame_size

    current_in = pivot_in + hop_size
    current_out = current_in + frame_size

    write_in = 0
    write_out = write_in + frame_size

    for f in range(num_frames):
        pivot_frame = copy(x[pivot_in:pivot_out])
        pivot_frame *= window

        current_frame = copy(x[current_in:current_out])
        current_frame *= window

        X_pivot = np.fft.rfft(pivot_frame)
        X_current = np.fft.rfft(current_frame)

        # From M. Puckette, "Phase-locked vocoder." 1995.
        tmp = Y_last
        tmp[1:] -= Y_last[:-1]
        tmp[:-1] -= Y_last[1:]
        Y_last = tmp

        Y_last[np.where(Y_last == 0)] = EPS
        X_pivot[np.where(X_pivot == 0)] = EPS

        a = Y_last / X_pivot
        b = abs(Y_last  / X_pivot)

        new_phase = a / b

        Y_current = X_current * new_phase

        y_current = np.fft.irfft(Y_current)
        y_current = np.real_if_close(y_current)
        y_current *= window

        y[write_in:write_out] += y_current

        # Store last frame.
        Y_last = Y_current

        # Advance pointers.
        pivot_in += analysis_hop_size
        pivot_out = pivot_in + frame_size

        current_in = pivot_in + hop_size
        current_out = current_in + frame_size

        write_in += hop_size
        write_out = write_in + frame_size

    # Truncate to nearest sample-length to match stretch value.
    num_samples_out = math.ceil(len(x) * stretch_factor)
    y = y[:num_samples_out]

    return y
