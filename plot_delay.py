import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mosqito.sq_metrics import loudness_zwtv
from mosqito.utils import load

from distance import calculate_distance, print_distance

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    pass

# Set up simple logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def extract_audio_from_video(video_path: Path) -> Optional[Path]:
    """Extract audio from the video and save it as a .wav file."""
    if 'VideoFileClip' not in globals():
        logging.warning("Warning: ffmpeg and moviepy must be installed to automatically extract audio.")
        return None

    audio_path = video_path.with_suffix(".wav")
    try:
        # Extract audio
        with VideoFileClip(str(video_path)) as video:
            audio = video.audio
            if audio is not None:
                audio.write_audiofile(str(audio_path), codec="pcm_s16le", logger=None)
            else:
                logging.error("No audio track found in the video.")
                return None
        
        logging.info("Audio extracted successfully!")
        return audio_path
    except Exception as e:
        logging.error(f"Error encountered during audio extraction: {e}")
        return None


def get_valid_file_path(prompt: str) -> Path:
    """Prompt the user for a file path until a valid existing file is provided."""
    while True:
        path_input = input(prompt).strip()
        
        # Clean quote marks from drag-and-drop actions
        path_input = path_input.strip('"').strip("'")
        
        path = Path(path_input)
        if path.is_file():
            logging.info(f"File found at {path}!")
            return path
        logging.error("File not found. Please check the path and try again.")


def process_audio(audio_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load audio and compute loudness along with the time axis."""
    logging.info("Processing audio...")
    sig, fs = load(str(audio_path), wav_calib=2 * 2**0.5)

    # Compute loudness
    # loudness_zwtv returns (loudness, N_spec, bark_axis, time)
    loudness, _, _, time_audio = loudness_zwtv(sig, fs, field_type="free")
    return loudness, time_audio


def process_video(video_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Extract red intensity from video frames and create a time axis."""
    logging.info("Processing video...")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    if frame_rate == 0:
        raise ValueError("Video has a frame rate of 0, cannot process.")

    logging.info(f"{int(frame_rate)} frames/second")
    
    red_intensity = np.zeros(nframes)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract the red channel (OpenCV uses BGR order)
        red_frame = frame[:, :, 2]
        
        # Take mean of the red channel across all pixels
        red_intensity[frame_index] = red_frame.mean()
        frame_index += 1

    cap.release()
    # Note: cv2.destroyAllWindows() was removed to prevent crashes in headless environments

    # Trim red_intensity to the actual number of frames read in case it's fewer than expected
    red_intensity = red_intensity[:frame_index]
    
    # Create time axis for video based on frame rate
    time_video = np.linspace(0.0, frame_index, frame_index, endpoint=False) / float(frame_rate)
    
    return red_intensity, time_video


def detect_flash_time(red_intensity: np.ndarray, time_video: np.ndarray) -> float:
    """Find the time of the flash using the largest increase in red intensity."""
    diff = np.diff(red_intensity)
    start_index = np.argmax(diff)
    return float(time_video[start_index])


def detect_boom_time(loudness: np.ndarray, time_audio: np.ndarray, flash_time: float) -> float:
    """Find the time of the boom immediately before the elevated loudness."""
    flash_idx = np.searchsorted(time_audio, flash_time)
    max_idx = int(np.argmax(loudness))
    
    if flash_idx < max_idx:
        peak = loudness[max_idx]
        threshold = peak * 0.1  # Start of explosion is typically around 10% of its absolute max peak
        
        # Search backwards from peak to find where it drops below the threshold
        base_idx = max_idx
        for i in range(max_idx, flash_idx, -1):
            if loudness[i] < threshold:
                base_idx = i
                break
                
        # Trace backwards a bit further to find the exact local minimum (the trough) before the rise
        for i in range(base_idx, flash_idx, -1):
            if loudness[i-1] > loudness[i]:
                return float(time_audio[i])
                
        return float(time_audio[base_idx])
    
    # Fallback if peak is before flash or overlapping 
    diff = np.diff(loudness)
    start_index = int(np.argmax(diff))
    return float(time_audio[start_index])


def plot_signals(
    time_audio: np.ndarray,
    loudness: np.ndarray,
    time_video: np.ndarray,
    red_intensity: np.ndarray,
    output_image_path: Path,
    flash_time: Optional[float] = None,
    boom_time: Optional[float] = None
) -> None:
    """Rescale signals and plot them together, saving to file and displaying."""
    # Rescale red intensity amplitude to match audio signal amplitude
    red_min = np.min(red_intensity)
    red_max = np.max(red_intensity)
    loudness_max = np.max(loudness)

    # Avoid division by zero if video was completely black or flat
    if red_max != red_min:
        red_intensity_scaled = (red_intensity - red_min) * (loudness_max / (red_max - red_min)) + np.min(loudness)
    else:
        red_intensity_scaled = red_intensity - red_min + np.min(loudness)

    # Get min and max of amplitude axis to prevent division by zero in y-limits
    amp_min = np.min(red_intensity_scaled)
    amp_max = np.max(red_intensity_scaled)
    amp_range = max(amp_max - amp_min, 1e-6)  # at least a small range

    y_min = amp_min - 0.1 * amp_range
    y_max = amp_max + 0.1 * amp_range

    # Plot
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(time_audio, loudness, "b-", label="Loudness")
    ax.plot(time_video, red_intensity_scaled, "r-", label="Red Intensity")
    
    if flash_time is not None:
        ax.axvline(x=flash_time, color='r', linestyle='--', alpha=0.7, label=f"Flash ({flash_time:.2f}s)")
    if boom_time is not None:
        ax.axvline(x=boom_time, color='b', linestyle='--', alpha=0.7, label=f"Boom ({boom_time:.2f}s)")

    ax.legend(loc="upper right")
    
    ax.set_xlim(time_audio[0], time_audio[-1])
    ax.set_ylim(y_min, y_max)
    ax.set_yticks([])
    ax.set_xlabel("Time [seconds]")
    ax.set_ylabel("Amplitude")
    plt.grid(True)

    # Save and show
    plt.savefig(str(output_image_path))
    logging.info(f"Plot saved to: {output_image_path}")
    plt.show()


def main() -> None:
    # 1. Get User Input
    video_path = get_valid_file_path("Enter path of video file: ")

    # If audio extraction fails, ask for audio path
    audio_path = extract_audio_from_video(video_path)
    if not audio_path:
        logging.info("You can still provide your own manually extracted audio file in .wav format.")
        audio_path = get_valid_file_path("Enter path of audio file: ")

    # 2. Process Audio
    loudness, time_audio = process_audio(audio_path)

    # 3. Process Video
    red_intensity, time_video = process_video(video_path)

    # 4. Automated Detection
    flash_time = detect_flash_time(red_intensity, time_video)
    boom_time = detect_boom_time(loudness, time_audio, flash_time)
    logging.info(f"Automated detection found Flash at {flash_time:.2f}s and Boom at {boom_time:.2f}s.")

    # 5. Distance Calculation
    try:
        temperature_input = input("Enter local temperature in degrees C (default 20): ").strip()
        temperature = float(temperature_input) if temperature_input else 20.0
    except ValueError:
        logging.warning("Invalid temperature entered. Defaulting to 20.0 deg C.")
        temperature = 20.0

    distance, error = calculate_distance(flash_time, boom_time, temperature)
    
    print("\n--- Distance Calculation ---")
    print_distance(flash_time, boom_time, temperature, distance, error)
    print("----------------------------\n")

    # 6. Plot Signals
    output_image_path = video_path.with_suffix(".png")
    plot_signals(time_audio, loudness, time_video, red_intensity, output_image_path, flash_time, boom_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
