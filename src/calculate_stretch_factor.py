def calculate_stretch_factor(array_length_samples, overlap_ms, sr):
    """Determine stretch factor to add `overlap_ms` to length of signal."""
    length_ms = array_length_samples / sr * 1000
    return (length_ms + overlap_ms) / length_ms
