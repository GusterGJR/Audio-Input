import pyaudio
import numpy as np
import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def one_channel(audio_data):
    np_audio = np.frombuffer(audio_data, dtype=np.int16)
    torch_audio = torch.tensor(np_audio, dtype=torch.float32, device=device)
    processed_audio = torch_audio * 2 # Apply a simple gain increase (multiply by)
    output_audio = processed_audio.cpu().numpy().astype(np.int16) # Move back to CPU and convert to NumPy

    return output_audio

def two_channel(audio_data):
    # Convert to NumPy array and reshape for stereo (2 channels)
    np_audio = np.frombuffer(audio_data, dtype=np.int16).reshape(-1, 2)  # Shape: (CHUNK, 2)
    torch_audio = torch.tensor(np_audio, dtype=torch.float32, device=device) # Convert to Torch tensor and move to GPU

    # Split into Left and Right channels
    left_channel = torch_audio[:, 0]  # Left channel
    right_channel = torch_audio[:, 1]  # Right channel
    # Example Processing: Apply different effects to each channel
    left_channel = left_channel * 1  # Increase volume of left channel, volume of left channel multiplied by
    right_channel = right_channel * 1  # Reduce volume of right channel, volume of right channel multiplied by

    diff = (left_channel - right_channel) / 2  # Create a difference between the channels
    comm = (left_channel + right_channel) / 2  # Combine the channels
    COMM, DIFF = 0.2, 1
    left_channel = comm * COMM + diff * DIFF # Add the difference to the combined channels
    right_channel = comm * COMM - diff * DIFF # Subtract the difference from the combined channels
    # In theory, COMM should be lower and DIFF should be higher for a more pronounced effect
    # in the environment of one micro in use and another under electricity

    # Merge channels back together
    processed_audio = torch.stack([left_channel, right_channel], dim=1)  # Shape: (CHUNK, 2)
    output_audio = processed_audio.cpu().numpy().astype(np.int16) # Move back to CPU and convert to NumPy

    return output_audio

if __name__ == "__main__":
    p = pyaudio.PyAudio()

    # List all available audio devices
    devices = []
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            devices.append(i)
            print(f"Device {i}: {device_info['name']} - Input Channels: {device_info['maxInputChannels']}")

    index = eval(input("Choose the audio device and press Enter..."))
    if index not in devices:
        print("Invalid device index. Please run the script again.")
        exit()

    # Audio Settings
    FORMAT = pyaudio.paInt16  # 16-bit audio, 32-bit seems too slow
    CHANNELS = p.get_device_info_by_index(index)['maxInputChannels']  # (Left + Right)
    RATE = 44100  # Sampling rate
    CHUNK = 1024  # Buffer size (Lower for lower latency)

    # Open input/output audio streams
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    input_device_index=index,  # Change this to the desired input device, if no external, choose 0/1
                    frames_per_buffer=CHUNK)

    print("Real-time audio loop started (Press Ctrl+C to stop)")

    try:
        while True:
            audio_data = stream.read(CHUNK) # Read microphone input
            
            if CHANNELS == 1:
                output_audio = one_channel(audio_data)
            elif CHANNELS == 2:
                output_audio = two_channel(audio_data)

            stream.write(output_audio.tobytes()) # Output to speakers

    except KeyboardInterrupt:
        print("Keyboard Interrupted...")

    finally:
        # Close stream and PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Audio loop stopped.")