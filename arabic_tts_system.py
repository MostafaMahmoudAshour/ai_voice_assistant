"""
Arabic Text-to-Speech (TTS) System
Supports multiple TTS engines with Arabic language support
"""

from pathlib import Path
from datetime import datetime

# TTS Engine 1: pyttsx3 (Offline, uses system voices)
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("pyttsx3 not available. Install with: pip install pyttsx3")

# TTS Engine 2: gTTS (Google TTS, requires internet)
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("gTTS not available. Install with: pip install gtts")

# TTS Engine 3: Edge TTS (Microsoft Edge TTS, requires internet, best quality)
try:
    import edge_tts
    import asyncio
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("edge-tts not available. Install with: pip install edge-tts")

# For playing audio
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not available. Install with: pip install pygame")


class ArabicTTS:
    """Text-to-Speech for Arabic with multiple engine support"""
    
    def __init__(self, engine="edge", voice=None):
        """
        Initialize TTS
        
        Args:
            engine: "edge" (best), "gtts" (good), "pyttsx3" (offline)
            voice: Specific voice ID (optional)
        """
        self.engine_name = engine
        self.voice = voice
        
        print(f"Initializing TTS with engine: {engine}")
        
        if engine == "pyttsx3":
            if not PYTTSX3_AVAILABLE:
                raise ImportError("pyttsx3 not installed")
            self.engine = pyttsx3.init()
            self._setup_pyttsx3()
        
        elif engine == "gtts":
            if not GTTS_AVAILABLE:
                raise ImportError("gTTS not installed")
            print("✓ gTTS ready (requires internet)")
        
        elif engine == "edge":
            if not EDGE_TTS_AVAILABLE:
                raise ImportError("edge-tts not installed")
            print("✓ Edge TTS ready (requires internet)")
        
        else:
            raise ValueError(f"Unknown engine: {engine}")
        
        # Initialize audio player
        if PYGAME_AVAILABLE:
            pygame.mixer.init()
            print("✓ Audio player ready")
    
    def _setup_pyttsx3(self):
        """Configure pyttsx3 engine"""
        # Set properties
        self.engine.setProperty('rate', 150)    # Speed (words per minute)
        self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        
        # List available voices
        voices = self.engine.getProperty('voices')
        
        print("\nAvailable pyttsx3 voices:")
        arabic_voices = []
        for i, voice in enumerate(voices):
            print(f"{i}: {voice.name} - {voice.languages}")
            if 'arabic' in voice.name.lower() or 'ar' in str(voice.languages).lower():
                arabic_voices.append(voice)
        
        # Try to set Arabic voice
        if self.voice:
            self.engine.setProperty('voice', self.voice)
        elif arabic_voices:
            self.engine.setProperty('voice', arabic_voices[0].id)
            print(f"\n✓ Using Arabic voice: {arabic_voices[0].name}")
        else:
            print("\n⚠️ No Arabic voice found, using default")
    
    def list_voices(self):
        """List all available voices for current engine"""
        if self.engine_name == "pyttsx3":
            voices = self.engine.getProperty('voices')
            print("\n" + "="*60)
            print("PYTTSX3 VOICES:")
            print("="*60)
            for i, voice in enumerate(voices):
                print(f"{i}: {voice.name}")
                print(f"   ID: {voice.id}")
                print(f"   Languages: {voice.languages}")
                print("-"*60)
        
        elif self.engine_name == "edge":
            print("\n" + "="*60)
            print("EDGE TTS VOICES")
            print("="*60)
            print("\nArabic Voices:")
            print("-"*60)
            arabic_voices = {
                "ar-EG-SalmaNeural": "Salma (Egyptian, Female) ⭐",
                "ar-EG-ShakirNeural": "Shakir (Egyptian, Male)",
                "ar-SA-HamedNeural": "Hamed (Saudi, Male)",
                "ar-SA-ZariyahNeural": "Zariyah (Saudi, Female)",
                "ar-AE-FatimaNeural": "Fatima (UAE, Female)",
                "ar-AE-HamdanNeural": "Hamdan (UAE, Male)",
                "ar-SY-AmanyNeural": "Amany (Syrian, Female)",
                "ar-SY-LaithNeural": "Laith (Syrian, Male)",
            }
            for voice_id, description in arabic_voices.items():
                print(f"{voice_id}: {description}")
            
            print("\nEnglish Voices (US):")
            print("-"*60)
            english_voices = {
                "en-US-AriaNeural": "Aria (Female) ⭐",
                "en-US-GuyNeural": "Guy (Male)",
                "en-US-JennyNeural": "Jenny (Female)",
                "en-US-ChristopherNeural": "Christopher (Male)",
            }
            for voice_id, description in english_voices.items():
                print(f"{voice_id}: {description}")
            print("="*60)
        
        elif self.engine_name == "gtts":
            print("\n" + "="*60)
            print("GTTS VOICES:")
            print("="*60)
            print("ar: Arabic")
            print("en: English")
            print("="*60)
    
    def speak(self, text, save_to_file=None, play_audio=True):
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            save_to_file: Path to save audio file (optional)
            play_audio: Whether to play the audio immediately
        
        Returns:
            Path to audio file if saved
        """
        print(f"\n🔊 Speaking: '{text[:50]}...'")
        
        if self.engine_name == "pyttsx3":
            return self._speak_pyttsx3(text, save_to_file, play_audio)
        elif self.engine_name == "gtts":
            return self._speak_gtts(text, save_to_file, play_audio)
        elif self.engine_name == "edge":
            return self._speak_edge(text, save_to_file, play_audio)
    
    def _speak_pyttsx3(self, text, save_to_file, play_audio):
        """Speak using pyttsx3 (offline)"""
        if save_to_file:
            self.engine.save_to_file(text, save_to_file)
            self.engine.runAndWait()
            print(f"💾 Saved to: {save_to_file}")
            if play_audio:
                self._play_audio(save_to_file)
            return save_to_file
        else:
            if play_audio:
                self.engine.say(text)
                self.engine.runAndWait()
            return None
    
    def _speak_gtts(self, text, save_to_file, play_audio):
        """Speak using Google TTS (requires internet)"""
        # Generate audio
        tts = gTTS(text=text, lang='ar', slow=False)
        
        # Save to file
        if not save_to_file:
            save_to_file = "temp_tts_output.mp3"
        
        tts.save(save_to_file)
        print(f"💾 Saved to: {save_to_file}")
        
        # Play audio
        if play_audio:
            self._play_audio(save_to_file)
        
        return save_to_file
    
    def _speak_edge(self, text, save_to_file, play_audio):
        """Speak using Edge TTS (requires internet, best quality)"""
        # Use default Egyptian voice if not specified
        voice = self.voice or "ar-EG-SalmaNeural"
        
        if not save_to_file:
            save_to_file = "temp_tts_output.mp3"
        
        # Run async function
        asyncio.run(self._edge_tts_async(text, voice, save_to_file))
        
        print(f"💾 Saved to: {save_to_file}")
        
        # Play audio
        if play_audio:
            self._play_audio(save_to_file)
        
        return save_to_file
    
    async def _edge_tts_async(self, text, voice, output_file):
        """Async function for Edge TTS"""
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
    
    def _play_audio(self, audio_file):
        """Play audio file using pygame"""
        if not PYGAME_AVAILABLE:
            print("⚠️ pygame not available, cannot play audio")
            print(f"   Play manually: {audio_file}")
            return
        
        try:
            print("▶️ Playing audio...")
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for audio to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            print("✓ Audio finished")
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
    
    def speak_async(self, text, save_to_file=None):
        """
        Speak without blocking (useful for real-time responses)
        """
        import threading
        thread = threading.Thread(
            target=self.speak, 
            args=(text, save_to_file, True)
        )
        thread.start()
        return thread
    
    def set_voice(self, voice_id):
        """Change voice"""
        self.voice = voice_id
        if self.engine_name == "pyttsx3":
            self.engine.setProperty('voice', voice_id)
            print(f"✓ Voice changed to: {voice_id}")
    
    def set_speed(self, rate):
        """
        Change speaking speed
        
        Args:
            rate: Speed in words per minute (typically 100-200)
        """
        if self.engine_name == "pyttsx3":
            self.engine.setProperty('rate', rate)
            print(f"✓ Speed changed to: {rate} WPM")
    
    def set_volume(self, volume):
        """
        Change volume
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if self.engine_name == "pyttsx3":
            self.engine.setProperty('volume', volume)
            print(f"✓ Volume changed to: {volume}")


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("ARABIC TEXT-TO-SPEECH SYSTEM - Testing")
    print("="*60)
    
    # Test phrases
    # test_phrases = [
    #     "hello, how are you today?",
    #     "Is it all okay.",
    #     "What about weather today in Cairo?"
    # ]
    test_phrases = [
        "مرحبا، كيف حالك؟",
        "أنا مساعد صوتي يتحدث العربية",
        "الطقس اليوم جميل في القاهرة",
    ]
    
    # Method 1: Edge TTS (Best quality, requires internet) ⭐ RECOMMENDED
    print("\n" + "="*60)
    print("Testing Edge TTS (Recommended)")
    print("="*60)
    try:
        tts_edge = ArabicTTS(engine="edge", voice="ar-EG-SalmaNeural")
        tts_edge.list_voices()
        
        for i, phrase in enumerate(test_phrases):
            print(f"\nTest {i+1}:")
            tts_edge.speak(phrase, save_to_file=f"edge_output_{i+1}.mp3")
            print("-"*60)
    except Exception as e:
        print(f"Edge TTS error: {e}")
    
    # Method 2: gTTS (Good quality, requires internet)
    print("\n" + "="*60)
    print("Testing Google TTS")
    print("="*60)
    try:
        tts_gtts = ArabicTTS(engine="gtts")
        tts_gtts.speak(test_phrases[0], save_to_file="gtts_output.mp3")
    except Exception as e:
        print(f"gTTS error: {e}")
    
    # Method 3: pyttsx3 (Offline, may have limited Arabic support)
        print("\n" + "="*60)
        print("Testing pyttsx3 (Offline)")
        print("="*60)
        try:
            tts_pyttsx3 = ArabicTTS(engine="pyttsx3")
            tts_pyttsx3.list_voices()
            tts_pyttsx3.speak(test_phrases, save_to_file="pyttsx3_output.wav")
        except Exception as e:
            print(f"pyttsx3 error: {e}")

        print("\n✓ TTS Testing Complete!")
        print("\nRecommendation: Use Edge TTS for best Arabic quality!")