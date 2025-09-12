# audio_processing_pa_ta_ka
 This project contains a Python script to count the number of times the syllable sequence "Pa Ta Ka" is said in an audio file.  ## Description  The script `pa_ta_ka_counter.py` takes an audio file as input, transcribes the speech to text, and then counts the occurrences of the complete phrase "Pa Ta Ka". 


Syllable Sequence Counter - Analysis Methodology and Limitations

Overview

This tool analyzes audio files to automatically count syllable sequences (specifically "Pa-Ta-Ka" triplets) using advanced signal processing techniques. The current implementation represents the most accurate automated approach developed through extensive testing and optimization.

Installation and Usage

Requirements
pip install -r requirements.txt


Run the syllable counter:

python syllable_counter.py <your_audio_file.wav>
here it is 

python syllable_counter.py ptk_2.wav


Analysis Methodology

1. Audio Preprocessing Pipeline

    Time-Stretching (Speed Factor: 0.85x)

        • Purpose: Slow down audio by 15% to better separate rapid or overlapping syllables

        • Method: Overlap-add time-stretching with pitch preservation

        • Rationale: Rapid speech can cause syllable boundaries to blur, leading to missed detections

    Denoising Chain

        1.High-pass filtering (80 Hz cutoff) - Removes low-frequency noise and rumble

        2.Median filtering (5ms kernel) - Eliminates impulse noise and clicks

        3.Notch filtering (50/60/100/120 Hz) - Removes electrical interference

        4.Adaptive noise gate - Suppresses background noise between syllables

        5.Speech bandpass (150-3500 Hz) - Focuses on speech frequency range


2. Syllable Detection Algorithm

    Multi-Envelope Analysis

    The system uses three complementary envelope extraction methods:

        1.Hilbert Transform Envelope

            •Captures instantaneous amplitude

            •Smoothing: 15ms Gaussian filter

            •Best for: Overall syllable energy detection



        2.RMS Envelope (30ms window)

            •Root Mean Square energy calculation

            •Smoothing: 10ms Gaussian filter

            •Best for: Consistent energy-based detection



        3.Short-term Energy (15ms window)

            •Rapid energy changes

            •Smoothing: 8ms Gaussian filter

            •Best for: Sharp onset detection



    Adaptive Peak Detection

        •Window-based analysis: 1.5-second sliding windows

        •Local thresholding: Mean + 0.3 × Standard Deviation

        •Minimum distance: 40ms between peaks (prevents double-detection)

        •Prominence requirement: 0.1 × Local Standard Deviation

    Consensus Voting

        •Tolerance: 40ms spatial tolerance for peak alignment

        •Minimum votes: Requires agreement from at least 2 of 3 envelope methods

        •Position: Uses median position of agreeing peaks

3. Sequence Grouping

    Gap-based Clustering

    Timing Validation

    For each potential 3-syllable sequence:

        •Total duration: 0.24s - 4.12s (adjusted for 0.85x speed)

        •Average gap: 0.047s - 1.76s between syllables

        •Gap consistency: Standard deviation ≤ 0.94s

    Duplicate Removal

        •Spatial tolerance: 150ms (accounts for slowed audio timing)

        •Overlap detection: Sequences sharing 2+ peaks within tolerance are merged

4. Time Scale Conversion

    •Converts slowed audio timestamps back to original time scale

    •Multiplies all times by speed factor (0.85) for accurate reporting



Why 23 vs Target 27?

The 4-sequence difference (15% under-detection) represents the current limitation of automated detection. This gap occurs due to several factors detailed in the limitations section below.


