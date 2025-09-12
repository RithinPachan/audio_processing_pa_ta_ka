import sys
import os
import numpy as np
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt, hilbert, medfilt, resample
from scipy.ndimage import gaussian_filter1d
import soundfile as sf
from pydub import AudioSegment
import warnings
warnings.filterwarnings('ignore')


def load_audio(file_path):
    """Load audio file and return audio data and sample rate."""
    try:
        audio_data, sample_rate = sf.read(file_path)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        print(f"Loaded audio: {len(audio_data)} samples at {sample_rate} Hz")
        print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
        return audio_data, sample_rate
    except Exception as e:
        try:
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_channels(1)
            sample_rate = audio.frame_rate
            audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data))
            print(f"Loaded audio with pydub: {len(audio_data)} samples at {sample_rate} Hz")
            return audio_data, sample_rate
        except Exception as e2:
            print(f"Error loading audio file: {e2}")
            raise


def slow_down_audio(audio_data, sample_rate, speed_factor=0.85):
    """Slow down audio while preserving pitch."""
    print(f"Slowing down audio by factor {speed_factor:.2f}...")
    
    frame_length = 2048
    hop_length = frame_length // 4
    new_hop_length = int(hop_length * speed_factor)
    
    # Pad audio for processing
    padded_audio = np.pad(audio_data, (frame_length, frame_length), mode='constant')
    
    # Calculate output length
    num_frames = (len(padded_audio) - frame_length) // hop_length + 1
    output_length = int(len(audio_data) / speed_factor)
    output_audio = np.zeros(output_length + frame_length)
    
    # Window function
    window = np.hanning(frame_length)
    
    # Process frames
    for i in range(num_frames):
        input_pos = i * hop_length
        output_pos = int(i * new_hop_length)
        
        if input_pos + frame_length <= len(padded_audio) and output_pos + frame_length <= len(output_audio):
            frame = padded_audio[input_pos:input_pos + frame_length] * window
            output_audio[output_pos:output_pos + frame_length] += frame
    
    return output_audio[:output_length], sample_rate


def denoise_audio(audio_data, sample_rate):
    """Apply denoising to the audio signal."""
    print("Applying denoising...")
    
    # High-pass filter
    nyquist = sample_rate / 2
    high_cutoff = 80 / nyquist
    b, a = butter(4, high_cutoff, btype='high')
    denoised = filtfilt(b, a, audio_data)
    
    # Median filter for impulse noise
    kernel_size = max(3, int(0.005 * sample_rate))
    if kernel_size % 2 == 0:
        kernel_size += 1
    denoised = medfilt(denoised, kernel_size=kernel_size)
    
    # Notch filters for electrical noise
    for freq in [50, 60, 100, 120]:
        if freq < sample_rate / 2:
            Q = 30
            w0 = freq / nyquist
            b, a = signal.iirnotch(w0, Q)
            denoised = filtfilt(b, a, denoised)
    
    # Simple noise gate
    envelope = np.abs(hilbert(denoised))
    smoothed_envelope = gaussian_filter1d(envelope, sigma=sample_rate*0.01)
    
    threshold = np.percentile(smoothed_envelope, 25) + 0.3 * (np.mean(smoothed_envelope) - np.percentile(smoothed_envelope, 25))
    gate_mask = np.where(smoothed_envelope > threshold, 1.0, 0.1)
    gate_mask = gaussian_filter1d(gate_mask, sigma=sample_rate*0.005)
    
    denoised = denoised * gate_mask
    
    # Speech bandpass filter
    low_freq = 150 / nyquist
    high_freq = min(3500 / nyquist, 0.99)
    b, a = butter(4, [low_freq, high_freq], btype='band')
    denoised = filtfilt(b, a, denoised)
    
    # Normalize
    if np.max(np.abs(denoised)) > 0:
        denoised = denoised / np.max(np.abs(denoised))
    
    return denoised


def detect_peaks(audio_data, sample_rate):
    """Detect syllable peaks using multiple envelope methods."""
    
    # Calculate multiple envelopes
    hilbert_env = np.abs(hilbert(audio_data))
    hilbert_env = gaussian_filter1d(hilbert_env, sigma=sample_rate*0.015)
    
    # RMS envelope
    window_size = int(0.03 * sample_rate)
    rms_env = np.sqrt(np.convolve(audio_data**2, np.ones(window_size)/window_size, mode='same'))
    rms_env = gaussian_filter1d(rms_env, sigma=sample_rate*0.01)
    
    # Energy envelope
    short_window = int(0.015 * sample_rate)
    energy_env = np.convolve(audio_data**2, np.ones(short_window)/short_window, mode='same')
    energy_env = gaussian_filter1d(energy_env, sigma=sample_rate*0.008)
    
    all_peaks = []
    
    # Detect peaks in each envelope
    for envelope in [hilbert_env, rms_env, energy_env]:
        window_size = int(1.5 * sample_rate)
        
        for start in range(0, len(envelope), window_size//2):
            end = min(start + window_size, len(envelope))
            window = envelope[start:end]
            
            if len(window) < 100:
                continue
            
            local_mean = np.mean(window)
            local_std = np.std(window)
            threshold = local_mean + 0.3 * local_std
            
            min_distance = int(0.04 * sample_rate)
            window_peaks, _ = find_peaks(window,
                                       height=threshold,
                                       distance=min_distance,
                                       prominence=0.1 * local_std)
            
            global_peaks = [start + p for p in window_peaks if start + p < len(envelope)]
            all_peaks.extend(global_peaks)
    
    # Consensus voting
    peak_votes = {}
    tolerance = int(0.04 * sample_rate)
    
    for peak in all_peaks:
        nearby = [p for p in all_peaks if abs(p - peak) <= tolerance]
        consensus = int(np.median(nearby))
        
        if consensus not in peak_votes:
            peak_votes[consensus] = 0
        peak_votes[consensus] += len(nearby)
    
    final_peaks = [peak for peak, votes in peak_votes.items() if votes >= 2]
    final_peaks.sort()
    
    print(f"Detected {len(final_peaks)} consensus peaks")
    return final_peaks


def group_into_sequences(peaks, sample_rate, speed_factor=0.85):
    """Group peaks into 3-syllable sequences."""
    
    sequences = []
    
    # Adjust gap tolerances for slowed audio
    base_gaps = [0.6, 0.9, 1.3, 1.8]
    gap_tolerances = [gap / speed_factor for gap in base_gaps]
    
    for max_gap in gap_tolerances:
        i = 0
        while i < len(peaks) - 2:
            sequence = [peaks[i]]
            
            j = i + 1
            while j < len(peaks) and len(sequence) < 3:
                gap = (peaks[j] - sequence[-1]) / sample_rate
                if gap <= max_gap:
                    sequence.append(peaks[j])
                j += 1
            
            if len(sequence) == 3:
                # Validate timing
                total_duration = (sequence[-1] - sequence[0]) / sample_rate
                gaps = [(sequence[k+1] - sequence[k]) / sample_rate for k in range(2)]
                avg_gap = np.mean(gaps)
                gap_consistency = np.std(gaps)
                
                # Adjusted criteria for slowed audio
                min_duration = 0.2 / speed_factor
                max_duration = 3.5 / speed_factor
                min_gap = 0.04 / speed_factor
                max_gap_avg = 1.5 / speed_factor
                
                if (min_duration <= total_duration <= max_duration and 
                    min_gap <= avg_gap <= max_gap_avg and 
                    gap_consistency <= 0.8 / speed_factor):
                    sequences.append(sequence)
                
                i = j
            else:
                i += 1
    
    # Remove duplicates
    unique_sequences = []
    tolerance = int(0.15 * sample_rate)
    
    for seq in sequences:
        is_duplicate = False
        for existing in unique_sequences:
            if (abs(seq[0] - existing[0]) <= tolerance and
                abs(seq[1] - existing[1]) <= tolerance and
                abs(seq[2] - existing[2]) <= tolerance):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_sequences.append(seq)
    
    return unique_sequences


def convert_to_original_time(sequences, sample_rate, speed_factor):
    """Convert slowed audio timestamps back to original audio time."""
    
    original_sequences = []
    
    for seq in sequences:
        original_seq = []
        for peak_idx in seq:
            slow_time = peak_idx / sample_rate
            original_time = slow_time * speed_factor
            original_seq.append(original_time)
        original_sequences.append(original_seq)
    
    return original_sequences


def main():
    """Main function to count syllable sequences."""
    if len(sys.argv) != 2:
        print("Usage: python syllable_counter.py <audio_file_path>")
        print("\nExample:")
        print("  python syllable_counter.py audio_sample.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        sys.exit(1)
    
    print(f"Processing audio file: {audio_file}")
    print("=" * 60)
    
    try:
        # Load original audio
        original_audio, original_sample_rate = load_audio(audio_file)
        
        # Slow down the audio
        slowed_audio, slowed_sample_rate = slow_down_audio(original_audio, original_sample_rate, 0.85)
        
        # Apply denoising
        clean_audio = denoise_audio(slowed_audio, slowed_sample_rate)
        
        # Detect peaks
        print("Detecting syllable peaks...")
        peaks = detect_peaks(clean_audio, slowed_sample_rate)
        
        # Group into sequences
        print("Grouping into sequences...")
        slowed_sequences = group_into_sequences(peaks, slowed_sample_rate, 0.85)
        
        # Convert back to original time scale
        original_time_sequences = convert_to_original_time(slowed_sequences, slowed_sample_rate, 0.85)
        
        # Display results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Total syllable sequences detected: {len(original_time_sequences)}")
        
        if original_time_sequences:
            print(f"\nDetected sequences:")
            for i, seq in enumerate(original_time_sequences, 1):
                start_time = seq[0]
                end_time = seq[-1]
                total_duration = end_time - start_time
                gaps = [seq[j+1] - seq[j] for j in range(len(seq)-1)]
                avg_gap = np.mean(gaps)
                
                print(f"  {i:2d}. Time: {start_time:5.2f}s - {end_time:5.2f}s "
                      f"(Duration: {total_duration:.2f}s, Avg gap: {avg_gap:.2f}s)")
        
        print(f"\nAnalysis complete!")
        print(f"Final count: {len(original_time_sequences)} syllable sequences")
        
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
