#!/usr/bin/env python3
"""
Test script to measure CPU and memory usage of pre-recording buffer.
"""
import os
import sys
import time
import psutil
import threading
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import buffer class
load_dotenv()
from collections import deque
import pyaudio

# Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

class PreRecordingBuffer:
    """Continuously buffers audio in memory to capture pre-hotkey speech."""

    def __init__(self, duration: int = 2, sample_rate: int = SAMPLE_RATE, chunk_size: int = CHUNK_SIZE):
        self.duration = duration
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.enabled = duration > 0

        if not self.enabled:
            print("Pre-recording buffer disabled (duration = 0)")
            return

        # Calculate buffer size (number of chunks to keep)
        chunks_per_second = sample_rate / chunk_size
        self.max_chunks = int(chunks_per_second * duration)

        # Circular buffer for audio chunks
        self.buffer = deque(maxlen=self.max_chunks)
        self.is_running = False
        self.thread = None
        self.audio = None
        self.stream = None

        print(f"Pre-recording buffer initialized: {duration}s ({self.max_chunks} chunks)")

    def start(self):
        """Start continuous background recording."""
        if not self.enabled or self.is_running:
            return

        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            self.is_running = True
            self.thread = threading.Thread(target=self._record_loop, daemon=True)
            self.thread.start()
            print("Pre-recording buffer started")
        except Exception as e:
            print(f"Failed to start pre-recording buffer: {e}")
            self.enabled = False

    def _record_loop(self):
        """Continuously record audio into circular buffer."""
        while self.is_running:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.buffer.append(data)
            except Exception as e:
                print(f"Error in pre-recording buffer: {e}")
                break

    def get_buffered_audio(self):
        """Get all buffered audio as bytes."""
        if not self.enabled or not self.buffer:
            return b''
        return b''.join(self.buffer)

    def stop(self):
        """Stop the buffer and cleanup resources."""
        if not self.enabled:
            return

        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if self.audio:
            self.audio.terminate()

        print("Pre-recording buffer stopped")


def measure_usage(duration: int = 10):
    """Measure CPU and memory usage over time."""
    process = psutil.Process()

    # Get baseline measurements (before starting buffer)
    print("Measuring baseline (no buffer)...")
    time.sleep(2)
    baseline_cpu = process.cpu_percent(interval=1)
    baseline_mem = process.memory_info().rss / 1024 / 1024  # MB

    print(f"Baseline - CPU: {baseline_cpu:.1f}%, Memory: {baseline_mem:.2f} MB")
    print()

    # Start buffer
    print(f"Starting pre-recording buffer for {duration}s...")
    buffer = PreRecordingBuffer(duration=2)
    buffer.start()

    if not buffer.enabled:
        print("Buffer failed to start, exiting")
        return

    time.sleep(1)  # Let it stabilize

    # Measure over time
    print("\nMeasuring with buffer running:")
    print("Time | CPU%  | Memory (MB) | Buffer Size (KB)")
    print("-----|-------|-------------|------------------")

    for i in range(duration):
        cpu = process.cpu_percent(interval=1)
        mem = process.memory_info().rss / 1024 / 1024  # MB
        buffer_size = len(buffer.get_buffered_audio()) / 1024  # KB

        print(f"{i+1:4d}s | {cpu:5.1f} | {mem:11.2f} | {buffer_size:16.2f}")

    # Final measurements
    print()
    time.sleep(1)
    final_cpu = process.cpu_percent(interval=1)
    final_mem = process.memory_info().rss / 1024 / 1024
    final_buffer_size = len(buffer.get_buffered_audio()) / 1024

    print(f"\nFinal - CPU: {final_cpu:.1f}%, Memory: {final_mem:.2f} MB")
    print(f"Buffer is using: ~{final_buffer_size:.2f} KB")
    print(f"CPU increase: {final_cpu - baseline_cpu:.1f}%")
    print(f"Memory increase: {final_mem - baseline_mem:.2f} MB")

    # Cleanup
    buffer.stop()
    print("\nBuffer stopped")


if __name__ == "__main__":
    print("=== Pre-Recording Buffer Resource Usage Test ===\n")

    try:
        measure_usage(duration=10)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
