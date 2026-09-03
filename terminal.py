import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sounddevice as sd
import numpy as np
import queue
import threading
import time

# --- CONFIGURATION ---
INPUT_DEVICE_ID = 17   # CABLE Output (WASAPI)
OUTPUT_DEVICE_ID = 16  # Speakers (WASAPI)
SAMPLE_RATE = 48000
CHUNK_DURATION = 0.5 
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_DURATION)
WINDOW_CHUNKS = 3     

input_queue = queue.Queue()
output_queue = queue.Queue()

def audio_callback(indata, outdata, frames, time_info, status):
    if status:
        pass 
    
    input_queue.put(indata.copy())
    
    try:
        outdata[:] = output_queue.get_nowait()
    except queue.Empty:
        outdata[:] = np.zeros_like(indata)

def process_audio():
    # --- LOAD AI SAFELY INSIDE THE THREAD ---
    print("1. Loading AI Model (This takes a few seconds)...")
    from spleeter.separator import Separator
    separator = Separator('spleeter:2stems')

    print("2. Warming up AI...")
    dummy_audio = np.zeros((CHUNK_FRAMES * WINDOW_CHUNKS, 2))
    separator.separate(dummy_audio)
    print("-> AI Ready! Play music on Spotify.")

    rolling_buffer = []

    while True:
        chunk = input_queue.get()
        rolling_buffer.append(chunk)

        if len(rolling_buffer) < WINDOW_CHUNKS:
            continue
        
        window_audio = np.concatenate(rolling_buffer, axis=0)
        prediction = separator.separate(window_audio)
        instrumental = prediction['accompaniment']

        start_idx = CHUNK_FRAMES
        end_idx = CHUNK_FRAMES * 2
        safe_chunk = instrumental[start_idx:end_idx]

        output_queue.put(safe_chunk)
        rolling_buffer.pop(0)

if __name__ == "__main__":
    # Guarding the main execution prevents Windows multiprocessing crashes
    print("Starting system...")
    
    processing_thread = threading.Thread(target=process_audio, daemon=True)
    processing_thread.start()

    try:
        with sd.Stream(device=(INPUT_DEVICE_ID, OUTPUT_DEVICE_ID),
                       samplerate=SAMPLE_RATE, 
                       blocksize=CHUNK_FRAMES, 
                       channels=2, 
                       callback=audio_callback):
            
            print("Audio streams open. Press CTRL+C in this terminal to stop.")
            
            while True:
                time.sleep(1)
                
    except sd.PortAudioError as e:
        print(f"\n[CRITICAL AUDIO ERROR] {e}")
    except KeyboardInterrupt:
        print("\nShutting down karaoke machine...")