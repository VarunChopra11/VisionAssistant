#!/usr/bin/env python3
"""
Gemini Live API Python Client
Real-time audio and video interaction with Gemini 2.0 Live API
"""

import os
import asyncio
import json
import base64
import cv2
import pyaudio
import numpy as np
import websockets
import threading
import time
from typing import Optional, Callable
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiLiveClient:
    def __init__(self, api_key: str, project_id: str):
        # API Configuration
        self.api_key = api_key
        self.project_id = project_id
        self.model = "gemini-2.0-flash-live-preview-04-09"
        self.host = "us-central1-aiplatform.googleapis.com"
        self.service_url = f"wss://{self.host}/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
        self.model_uri = f"projects/{self.project_id}/locations/us-central1/publishers/google/models/{self.model}"
        
        # WebSocket connection
        self.websocket = None
        self.connected = False
        
        # Audio configuration
        self.audio_format = pyaudio.paInt16
        self.audio_channels = 1
        self.audio_rate = 16000
        self.audio_chunk = 1024
        self.audio_input = None
        self.audio_output = None
        self.audio_recording = False
        self.audio_playing = False
        
        # Video configuration
        self.video_capture = None
        self.video_streaming = False
        self.video_fps = 1  # 1 frame per second
        
        # Callbacks
        self.on_audio_response: Optional[Callable] = None
        self.on_text_response: Optional[Callable] = None
        self.on_connection_established: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Threading
        self.audio_input_thread = None
        self.audio_output_thread = None
        self.video_thread = None
        
        # Audio output buffer
        self.audio_output_buffer = asyncio.Queue()
        
    async def connect(self):
        """Establish WebSocket connection to Gemini Live API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            
            logger.info("Connecting to Gemini Live API...")
            self.websocket = await websockets.connect(
                self.service_url,
                additional_headers=headers
            )
            
            # Send initial setup message
            await self._send_setup_message()
            self.connected = True
            logger.info("Connected to Gemini Live API")
            
            if self.on_connection_established:
                self.on_connection_established()
                
            # Start message handling
            await self._handle_messages()
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            if self.on_error:
                self.on_error(f"Connection failed: {e}")
            raise
    
    async def _send_setup_message(self):
        """Send initial setup message to configure the session"""
        setup_message = {
            "setup": {
                "model": self.model_uri,
                "generation_config": {
                    "response_modalities": ["AUDIO", "TEXT"]
                },
                "system_instruction": {
                    "parts": [{
                        "text": "You are a helpful AI assistant capable of processing audio and video input. Respond naturally to user queries."
                    }]
                }
            }
        }
        
        await self.websocket.send(json.dumps(setup_message))
        logger.info("Setup message sent")
    
    async def _handle_messages(self):
        """Handle incoming messages from the API"""
        try:
            async for message in self.websocket:
                await self._process_message(json.loads(message))
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Error handling messages: {e}")
            if self.on_error:
                self.on_error(f"Message handling error: {e}")
    
    async def _process_message(self, data: dict):
        """Process incoming message from the API"""
        try:
            # Check for setup completion
            if data.get("setupComplete"):
                logger.info("Setup completed")
                return
            
            # Check for server content
            server_content = data.get("serverContent", {})
            model_turn = server_content.get("modelTurn", {})
            parts = model_turn.get("parts", [])
            
            for part in parts:
                # Handle text response
                if "text" in part:
                    text_content = part["text"]
                    logger.info(f"Received text: {text_content}")
                    if self.on_text_response:
                        self.on_text_response(text_content)
                
                # Handle audio response
                elif "inlineData" in part:
                    audio_data = part["inlineData"]["data"]
                    logger.info("Received audio response")
                    await self.audio_output_buffer.put(audio_data)
                    if self.on_audio_response:
                        self.on_audio_response(audio_data)
                        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def start_audio_input(self):
        """Start audio input capture"""
        if self.audio_recording:
            return
            
        try:
            self.audio_input = pyaudio.PyAudio()
            self.audio_recording = True
            
            self.audio_input_thread = threading.Thread(target=self._audio_input_worker)
            self.audio_input_thread.daemon = True
            self.audio_input_thread.start()
            
            logger.info("Audio input started")
        except Exception as e:
            logger.error(f"Failed to start audio input: {e}")
    
    def stop_audio_input(self):
        """Stop audio input capture"""
        self.audio_recording = False
        if self.audio_input_thread:
            self.audio_input_thread.join()
        if self.audio_input:
            self.audio_input.terminate()
            self.audio_input = None
        logger.info("Audio input stopped")
    
    def _audio_input_worker(self):
        """Worker thread for audio input"""
        try:
            stream = self.audio_input.open(
                format=self.audio_format,
                channels=self.audio_channels,
                rate=self.audio_rate,
                input=True,
                frames_per_buffer=self.audio_chunk
            )
            
            while self.audio_recording and self.connected:
                try:
                    audio_data = stream.read(self.audio_chunk, exception_on_overflow=False)
                    # Convert to base64
                    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                    
                    # Send audio to API
                    if self.websocket and not self.websocket.closed:
                        asyncio.run_coroutine_threadsafe(
                            self._send_audio_message(audio_b64),
                            asyncio.get_event_loop()
                        )
                except Exception as e:
                    logger.error(f"Audio input error: {e}")
                    break
            
            stream.close()
        except Exception as e:
            logger.error(f"Audio input worker error: {e}")
    
    async def _send_audio_message(self, audio_b64: str):
        """Send audio message to the API"""
        message = {
            "realtime_input": {
                "media_chunks": [{
                    "mime_type": "audio/pcm",
                    "data": audio_b64
                }]
            }
        }
        
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send audio message: {e}")
    
    def start_audio_output(self):
        """Start audio output playback"""
        if self.audio_playing:
            return
            
        try:
            self.audio_output = pyaudio.PyAudio()
            self.audio_playing = True
            
            self.audio_output_thread = threading.Thread(target=self._audio_output_worker)
            self.audio_output_thread.daemon = True
            self.audio_output_thread.start()
            
            logger.info("Audio output started")
        except Exception as e:
            logger.error(f"Failed to start audio output: {e}")
    
    def stop_audio_output(self):
        """Stop audio output playback"""
        self.audio_playing = False
        if self.audio_output_thread:
            self.audio_output_thread.join()
        if self.audio_output:
            self.audio_output.terminate()
            self.audio_output = None
        logger.info("Audio output stopped")
    
    def _audio_output_worker(self):
        """Worker thread for audio output"""
        try:
            stream = self.audio_output.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,  # Gemini outputs at 24kHz
                output=True,
                frames_per_buffer=1024
            )
            
            while self.audio_playing:
                try:
                    # Get audio data from buffer (with timeout)
                    future = asyncio.run_coroutine_threadsafe(
                        asyncio.wait_for(self.audio_output_buffer.get(), timeout=0.1),
                        asyncio.get_event_loop()
                    )
                    
                    try:
                        audio_b64 = future.result(timeout=0.1)
                        # Decode base64 audio
                        audio_data = base64.b64decode(audio_b64)
                        
                        # Convert PCM data for playback
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        
                        # Play audio
                        stream.write(audio_array.tobytes())
                        
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Audio playback error: {e}")
                        
                except Exception as e:
                    logger.error(f"Audio output error: {e}")
                    time.sleep(0.01)
            
            stream.close()
        except Exception as e:
            logger.error(f"Audio output worker error: {e}")
    
    def start_video_input(self, camera_index: int = 0):
        """Start video input capture"""
        if self.video_streaming:
            return
            
        try:
            self.video_capture = cv2.VideoCapture(camera_index)
            if not self.video_capture.isOpened():
                raise Exception("Failed to open camera")
                
            self.video_streaming = True
            
            self.video_thread = threading.Thread(target=self._video_input_worker)
            self.video_thread.daemon = True
            self.video_thread.start()
            
            logger.info("Video input started")
        except Exception as e:
            logger.error(f"Failed to start video input: {e}")
    
    def stop_video_input(self):
        """Stop video input capture"""
        self.video_streaming = False
        if self.video_thread:
            self.video_thread.join()
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        logger.info("Video input stopped")
    
    def _video_input_worker(self):
        """Worker thread for video input"""
        try:
            while self.video_streaming and self.connected:
                ret, frame = self.video_capture.read()
                if not ret:
                    logger.error("Failed to capture video frame")
                    break
                
                # Resize frame for efficiency
                frame = cv2.resize(frame, (640, 480))
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send frame to API
                if self.websocket and not self.websocket.closed:
                    asyncio.run_coroutine_threadsafe(
                        self._send_video_message(frame_b64),
                        asyncio.get_event_loop()
                    )
                
                # Control frame rate
                time.sleep(1.0 / self.video_fps)
                
        except Exception as e:
            logger.error(f"Video input worker error: {e}")
    
    async def _send_video_message(self, frame_b64: str):
        """Send video frame to the API"""
        message = {
            "realtime_input": {
                "media_chunks": [{
                    "mime_type": "image/jpeg",
                    "data": frame_b64
                }]
            }
        }
        
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send video message: {e}")
    
    async def send_text_message(self, text: str):
        """Send text message to the API"""
        message = {
            "client_content": {
                "turns": [{
                    "role": "user",
                    "parts": [{"text": text}]
                }],
                "turn_complete": True
            }
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            logger.info(f"Sent text message: {text}")
        except Exception as e:
            logger.error(f"Failed to send text message: {e}")
    
    def start_all_inputs(self):
        """Start all input streams (audio and video)"""
        self.start_audio_input()
        self.start_audio_output()
        self.start_video_input()
    
    def stop_all_inputs(self):
        """Stop all input streams"""
        self.stop_audio_input()
        self.stop_audio_output()
        self.stop_video_input()
    
    async def disconnect(self):
        """Disconnect from the API"""
        self.connected = False
        self.stop_all_inputs()
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        logger.info("Disconnected from Gemini Live API")


# Example usage and demo
async def demo():
    """Demo function showing how to use the GeminiLiveClient"""
    
    # REPLACE WITH YOUR ACTUAL API KEY AND PROJECT ID
    API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
    PROJECT_ID = os.getenv("GEMINI_PROJECT_ID", "YOUR_PROJECT_ID_HERE")
    
    client = GeminiLiveClient(API_KEY, PROJECT_ID)
    
    # Set up callbacks
    def on_text_response(text):
        print(f"🤖 Gemini: {text}")
    
    def on_audio_response(audio_data):
        print("🔊 Received audio response")
    
    def on_connection_established():
        print("✅ Connected to Gemini Live API")
        print("🎤 Starting audio and video input...")
        client.start_all_inputs()
    
    def on_error(error):
        print(f"❌ Error: {error}")
    
    # Register callbacks
    client.on_text_response = on_text_response
    client.on_audio_response = on_audio_response
    client.on_connection_established = on_connection_established
    client.on_error = on_error
    
    try:
        print("🚀 Starting Gemini Live API Demo")
        print("💡 Make sure you have:")
        print("   - A working microphone")
        print("   - A working camera")
        print("   - Valid API key and project ID")
        print()
        
        # Connect and run
        await client.connect()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        await client.disconnect()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        await client.disconnect()


if __name__ == "__main__":
    print("Gemini Live API Python Client")
    print("=" * 40)
    
    # Check dependencies
    try:
        import pyaudio
        import cv2
        import websockets
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install pyaudio opencv-python websockets numpy")
        exit(1)
    
    # Run demo
    asyncio.run(demo())


