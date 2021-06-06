import numpy as np


def apply_cross_fade(clips, cross_fade_ms, sr):
    """Concatenate audio clips with a cross fade."""

    num_clips = len(clips)

    cross_fade_samples = int(np.floor(cross_fade_ms * sr / 1000))
    fade_ramp = np.arange(cross_fade_samples) / cross_fade_samples

    # if not is_even(cross_fade_samples):
    #     cross_fade_samples += 1

    raw_num_samples = 0
    for clip in clips:
        raw_num_samples += len(clip)

    total_overlap_samples = (num_clips - 1) * cross_fade_samples
    num_samples = raw_num_samples - total_overlap_samples

    y = np.zeros(num_samples)
    write_in = 0
    for clip in clips:
        write_out = write_in + len(clip)

        # Update pointers.
        ramp_in = write_out - cross_fade_samples
        ramp_out = write_out

        # Fade in and place.
        clip[:cross_fade_samples] *= fade_ramp

        y[write_in:write_out] += clip

        # Fade out.
        y[ramp_in:ramp_out] *= (1 - fade_ramp)

        # Advance write pointer.
        write_in = ramp_in

    return y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.io.wavfile

    file_path = "../audio/008-you-possess-the-treasure-you-seek-seed001.wav"

    # Test audio file.
    sr, x = scipy.io.wavfile.read(file_path)
    x = x / np.iinfo(np.int16).max
    time_x = np.arange(len(x)) / sr

    plt.plot(time_x, x, label='Original')

    # Quick list-of-clips demo.
    tmp = []
    for i in range(20):
        tmp.append(x[i * 1000:(i + 1) * 1000])

    cross_fade_ms = 20
    y = apply_cross_fade(tmp, cross_fade_ms, sr)

    time_y = np.arange(len(y)) / sr
    plt.plot(time_y, y, label='Cross fade')
    plt.show()
