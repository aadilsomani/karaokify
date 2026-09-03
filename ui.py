import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sounddevice as sd
import numpy as np
import queue
import threading
import time
import webview

# Global state for controlling the audio stream thread, model cache, and active window reference
audio_thread = None
is_running = False
stop_event = threading.Event()
window_ref = None
separator = None  # Cache the model globally so it only loads once

class KaraokifyAPI:
    def get_devices(self):
        """Fetches available audio input and output devices for the dropdowns."""
        devices = sd.query_devices()
        inputs = []
        outputs = []
        
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                inputs.append({'id': idx, 'name': dev['name']})
            if dev['max_output_channels'] > 0:
                outputs.append({'id': idx, 'name': dev['name']})
                
        return {'inputs': inputs, 'outputs': outputs}

    def start_karaoke(self, input_id, output_id, chunk_duration, use_gpu):
        global audio_thread, is_running, stop_event
        if is_running:
            return {'status': 'error', 'message': 'Already running!'}

        input_id = int(input_id)
        output_id = int(output_id)
        chunk_duration = float(chunk_duration)
        use_gpu = bool(use_gpu)

        # Configure GPU / CPU visibility for TensorFlow prior to engine startup
        if use_gpu:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None) # Enable visible GPUs
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU-only mode

        stop_event.clear()
        is_running = True

        audio_thread = threading.Thread(
            target=run_karaoke_engine, 
            args=(input_id, output_id, chunk_duration, stop_event), 
            daemon=True
        )
        audio_thread.start()
        return {'status': 'success', 'message': 'Initializing...'}

    def stop_karaoke(self):
        global is_running, stop_event
        if not is_running:
            return {'status': 'error', 'message': 'Not running.'}
        
        stop_event.set()
        is_running = False
        return {'status': 'success', 'message': 'Stopped.'}

def update_ui_status(status_text, is_live=False):
    """Helper to push status updates from Python to the pywebview frontend."""
    if window_ref:
        safe_text = status_text.replace("'", "\\'")
        window_ref.evaluate_js(f"updateStatusFromPython('{safe_text}', {str(is_live).lower()});")

def run_karaoke_engine(input_id, output_id, chunk_duration, stop_event):
    global is_running, separator
    sample_rate = 48000
    chunk_frames = int(sample_rate * chunk_duration)
    window_chunks = 3

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    # Lazily load the model on first run only, preventing startup crashes
    if separator is None:
        update_ui_status("Loading AI Model into memory (First run may take a moment)...")
        from spleeter.separator import Separator
        try:
            separator = Separator('spleeter:2stems')
            dummy_audio = np.zeros((chunk_frames * window_chunks, 2))
            separator.separate(dummy_audio)
        except Exception as e:
            print(f"AI Load Error: {e}")
            update_ui_status(f"AI Load Error: {e}")
            is_running = False
            if window_ref:
                window_ref.evaluate_js("resetToStoppedState();")
            return

    update_ui_status("AI Model ready! Starting audio stream...", is_live=True)

    def audio_callback(indata, outdata, frames, time_info, status):
        input_queue.put(indata.copy())
        try:
            outdata[:] = output_queue.get_nowait()
        except queue.Empty:
            outdata[:] = np.zeros_like(indata)

    def process_loop():
        rolling_buffer = []
        while not stop_event.is_set():
            try:
                chunk = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            rolling_buffer.append(chunk)
            if len(rolling_buffer) < window_chunks:
                continue
            
            window_audio = np.concatenate(rolling_buffer, axis=0)
            prediction = separator.separate(window_audio)
            instrumental = prediction['accompaniment']

            start_idx = chunk_frames
            end_idx = chunk_frames * 2
            safe_chunk = instrumental[start_idx:end_idx]

            output_queue.put(safe_chunk)
            rolling_buffer.pop(0)

    processor = threading.Thread(target=process_loop, daemon=True)
    processor.start()

    try:
        with sd.Stream(device=(input_id, output_id),
                       samplerate=sample_rate, 
                       blocksize=chunk_frames, 
                       channels=2, 
                       callback=audio_callback):
            while not stop_event.is_set():
                time.sleep(0.1)
    except Exception as e:
        print(f"Stream Error: {e}")
        update_ui_status(f"Stream Error: {e}")
    
    is_running = False
    if window_ref:
        window_ref.evaluate_js("resetToStoppedState();")

# HTML/CSS/JS Frontend UI
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Karaokify</title>
    <style>
        :root {
            --bg-color: #121212;
            --card-bg: #1e1e1e;
            --primary: #1db954;
            --primary-hover: #1ed760;
            --text-main: #ffffff;
            --text-muted: #b3b3b3;
            --border: #2a2a2a;
            --warning-bg: rgba(255, 183, 3, 0.1);
            --warning-border: #ffb703;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .container {
            width: 100%;
            max-width: 480px;
        }
        .header {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 24px;
        }
        .logo {
            font-size: 32px;
            background: #2a2a2a;
            border: 1px solid #404040;
            border-radius: 12px;
            width: 54px;
            height: 54px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        h1 {
            margin: 0;
            font-size: 24px;
            font-weight: 700;
        }
        p.subtitle {
            margin: 4px 0 0 0;
            color: var(--text-muted);
            font-size: 13px;
        }
        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }
        .form-group {
            margin-bottom: 16px;
        }
        .form-group:last-child {
            margin-bottom: 0;
        }
        label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-muted);
        }
        select {
            width: 100%;
            padding: 10px 12px;
            background-color: #2a2a2a;
            border: 1px solid #404040;
            border-radius: 6px;
            color: var(--text-main);
            font-size: 14px;
            outline: none;
        }
        select:focus {
            border-color: var(--primary);
        }
        .slider-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .slider-value {
            font-weight: 600;
            color: var(--primary);
            font-size: 14px;
        }
        input[type="range"] {
            width: 100%;
            accent-color: var(--primary);
            cursor: pointer;
        }
        .checkbox-label {
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-main);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            margin-top: 4px;
        }
        input[type="checkbox"] {
            width: 16px;
            height: 16px;
            accent-color: var(--primary);
            cursor: pointer;
        }
        .warning-box {
            background-color: var(--warning-bg);
            border: 1px solid var(--warning-border);
            border-radius: 8px;
            padding: 12px;
            font-size: 12px;
            line-height: 1.4;
            color: #ffd166;
            margin-bottom: 20px;
        }
        .btn {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn-start {
            background-color: var(--primary);
            color: #000;
        }
        .btn-start:hover {
            background-color: var(--primary-hover);
        }
        .btn-stop {
            background-color: #e63946;
            color: #fff;
        }
        .btn-stop:hover {
            background-color: #d90429;
        }
        .status-indicator {
            text-align: center;
            margin-top: 12px;
            font-size: 13px;
            color: var(--text-muted);
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <div class="logo">🎧</div>
        <div>
            <h1>Karaokify</h1>
            <p class="subtitle">Real-time AI Karaoke Machine for Spotify</p>
        </div>
    </div>

    <div class="warning-box">
        <b>Note:</b> Starting the engine loads deep learning models into memory. The UI may freeze for 5–10 seconds on first launch while downloading/initializing.
    </div>

    <div class="card">
        <div class="form-group">
            <label for="inputDevice">Spotify Audio Source (Virtual Cable)</label>
            <select id="inputDevice"></select>
        </div>

        <div class="form-group">
            <label for="outputDevice">Playback Output (Speakers/Headphones)</label>
            <select id="outputDevice"></select>
        </div>

        <div class="form-group">
            <div class="slider-container">
                <label for="chunkDuration" style="margin:0;">Latency / Buffer Size</label>
                <span id="sliderVal" class="slider-value">0.5s (~1.0s delay)</span>
            </div>
            <input type="range" id="chunkDuration" min="0.10" max="1.50" step="0.01" value="0.50">
        </div>

        <div class="form-group" style="margin-top: 16px;">
            <label class="checkbox-label">
                <input type="checkbox" id="useGpu"> Enable GPU Acceleration (NVIDIA CUDA)
            </label>
        </div>
    </div>

    <button id="actionBtn" class="btn btn-start" onclick="toggleKaraoke()">Start Karaokify</button>
    <div id="statusText" class="status-indicator">Ready to initialize</div>
</div>

<script>
    let isRunning = false;

    window.addEventListener('pywebviewready', async function() {
        const devices = await window.pywebview.api.get_devices();
        
        const inputSelect = document.getElementById('inputDevice');
        const outputSelect = document.getElementById('outputDevice');

        devices.inputs.forEach(dev => {
            let opt = document.createElement('option');
            opt.value = dev.id;
            opt.text = dev.name;
            if (dev.name.toLowerCase().includes('cable output')) opt.selected = true;
            inputSelect.appendChild(opt);
        });

        devices.outputs.forEach(dev => {
            let opt = document.createElement('option');
            opt.value = dev.id;
            opt.text = dev.name;
            if (dev.name.toLowerCase().includes('speakers')) opt.selected = true;
            outputSelect.appendChild(opt);
        });
    });

    // Update latency helper text dynamically
    const slider = document.getElementById('chunkDuration');
    slider.addEventListener('input', (e) => {
        let val = parseFloat(e.target.value);
        let delay = (val * 2).toFixed(1);
        document.getElementById('sliderVal').innerText = `${val}s (~${delay}s delay)`;
    });

    async function toggleKaraoke() {
        const btn = document.getElementById('actionBtn');
        const status = document.getElementById('statusText');
        
        const inputId = document.getElementById('inputDevice').value;
        const outputId = document.getElementById('outputDevice').value;
        const chunkDuration = slider.value;
        const useGpu = document.getElementById('useGpu').checked;

        if (!isRunning) {
            btn.innerText = "Initializing...";
            status.innerText = "Triggering engine initialization...";
            
            const res = await window.pywebview.api.start_karaoke(inputId, outputId, chunkDuration, useGpu);
            
            if (res.status === 'success') {
                isRunning = true;
            } else {
                btn.innerText = "Start Karaokify";
                btn.className = "btn btn-start";
                status.innerText = "Error: " + res.message;
            }
        } else {
            await window.pywebview.api.stop_karaoke();
            resetToStoppedState();
        }
    }

    // Called dynamically from Python backend thread
    function updateStatusFromPython(message, isLive) {
        const status = document.getElementById('statusText');
        const btn = document.getElementById('actionBtn');
        status.innerText = message;
        
        if (isLive) {
            isRunning = true;
            btn.innerText = "Stop Karaokify";
            btn.className = "btn btn-stop";
            status.innerText = "Karaokify is live! Play music on Spotify.";
        }
    }

    // Resets button and UI text back to safe initial stopped state
    function resetToStoppedState() {
        isRunning = false;
        const btn = document.getElementById('actionBtn');
        const status = document.getElementById('statusText');
        btn.innerText = "Start Karaokify";
        btn.className = "btn btn-start";
        status.innerText = "Stopped.";
    }
</script>

</body>
</html>
"""
if __name__ == '__main__':
    api = KaraokifyAPI()
    window_ref = webview.create_window('Karaokify - AI Karaoke', html=html_content, js_api=api, width=540, height=690, resizable=False)
    webview.start()