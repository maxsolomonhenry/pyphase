import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile


def phase_vocoder(x, stretch_factor, frame_size=1024):
    """Time-stretch a signal x by a given stretch factor."""

    EPS = 1e-12

    hop_size = frame_size // 4
    hN = int(frame_size // 2) + 1
    window = np.hamming(frame_size)

    # Analysis hops faster or slower than synthesis, to time- squish or stretch.
    analysis_hop_size = int(hop_size // stretch_factor)

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
        pivot_frame = x[pivot_in:pivot_out]
        pivot_frame *= window

        current_frame = x[current_in:current_out]

        X_pivot = np.fft.rfft(pivot_frame)
        X_current = np.fft.rfft(current_frame)
        pivot_frame *= window

        # From M. Puckette, "Phase-locked vocoder." 1995.
        Y_phase_locked = Y_last
        Y_phase_locked[1:] -= Y_last[:-1]
        Y_phase_locked[:-1] -= Y_last[1:]

        tmp = (Y_phase_locked + EPS) / (X_pivot + EPS)
        new_phase = tmp / np.abs(tmp)

        Y_current = X_current * new_phase

        y_current = np.fft.irfft(Y_current)
        y_current = np.real(y_current)
        y_current *= (window)

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

    return y


if __name__ == '__main__':

    file_path = "../audio/008-you-possess-the-treasure-you-seek-seed001.wav"

    # Test audio file.
    sr, x = scipy.io.wavfile.read(file_path)
    x = x / np.iinfo(np.int16).max
    time_x = np.arange(len(x)) / sr

    plt.plot(time_x, x, label='Original')

    stretch_factor = 1
    y = phase_vocoder(x, stretch_factor, frame_size=512)

    time_y = np.arange(len(y)) / sr
    plt.plot(time_y, y, label='Phase vocoder')
    plt.show()
