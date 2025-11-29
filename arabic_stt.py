import whisper
import torch
import time
from pathlib import Path

class ArabicSTT:
    def __init__(self, model_size="small", device=None):
        """
        Initialize Whisper STT for Arabic
        
        Args:
            model_size: "tiny", "base", "small", "medium", "large"
            device: "cuda" or "cpu" (auto-detected if None)
        """
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Whisper {model_size} model on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print(f"Model loaded successfully!")
        
        # Check VRAM usage if using GPU
        if self.device == "cuda":
            vram_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"VRAM allocated: {vram_allocated:.2f} GB")
    
    def transcribe_file(self, audio_path, language="ar", task="transcribe", 
                       initial_prompt=None, temperature=0.0):
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            language: Language code ("ar" for Arabic)
            task: "transcribe" 
            initial_prompt: Hint text to guide the model (useful for Egyptian dialect)
            temperature: Sampling temperature (0 = deterministic, higher = more creative)
        
        Returns:
            dict with 'text', 'segments', and 'language'
        """
        start_time = time.time()
        
        print(f"\nTranscribing: {audio_path}")
        
        # Transcription options
        options = {
            "language": language,
            "task": task,
            "fp16": (self.device == "cuda"),
            "temperature": temperature,
        }
        
        # Add initial prompt if provided (helps with Egyptian dialect)
        if initial_prompt:
            options["initial_prompt"] = initial_prompt
            print(f"Using dialect hint: {initial_prompt[:50]}...")
        
        result = self.model.transcribe(audio_path, **options)
        
        elapsed = time.time() - start_time
        print(f"Transcription completed in {elapsed:.2f} seconds")
        
        return result
    
    def transcribe_egyptian(self, audio_path):
        """
        Optimized transcription for Egyptian Arabic dialect
        """
        # Egyptian Arabic prompt to guide the model
        egyptian_prompt = "إزيك، عامل إيه، ماشي، يعني، كده، علشان، عايز، ازاي"
        
        return self.transcribe_file(
            audio_path, 
            language="ar",
            initial_prompt=egyptian_prompt,
            temperature=0.0  # More deterministic for better accuracy
        )
    
    def transcribe_with_timestamps(self, audio_path, language="ar"):
        """
        Transcribe with detailed timestamps for each segment
        """
        result = self.transcribe_file(audio_path, language=language)
        
        print("\n" + "="*60)
        print("TRANSCRIPTION WITH TIMESTAMPS")
        print("="*60)
        
        for segment in result['segments']:
            start = segment['start']
            end = segment['end']
            text = segment['text']
            print(f"[{start:.2f}s -> {end:.2f}s] {text}")
        
        return result
    
    def save_transcription(self, result, output_path, include_timestamps=False):
        """
        Save transcription to text file with proper UTF-8 encoding for Arabic
        
        Args:
            result: Transcription result from whisper
            output_path: Output file path
            include_timestamps: If True, include timestamps for each segment
        """
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            if include_timestamps and 'segments' in result:
                for segment in result['segments']:
                    start = segment['start']
                    end = segment['end']
                    text = segment['text'].strip()
                    f.write(f"[{start:.2f}s -> {end:.2f}s] {text}\n")
            else:
                f.write(result['text'])
        print(f"\nTranscription saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize STT with Small model (recommended for basic hardware)
    stt = ArabicSTT(model_size="small")
    
    # Example 1: Simple transcription
    audio_file = "001.mp3"  # Replace with your audio file
    
    if Path(audio_file).exists():
        # Method 1: Standard Arabic transcription
        # result = stt.transcribe_file(audio_file, language="ar")
        
        # Method 2: Optimized for Egyptian Arabic (recommended for Egyptian accent)
        result = stt.transcribe_egyptian(audio_file)
        
        # Method 3: Custom prompt for your specific context
        # custom_prompt = "أهلا وسهلا، إزيك، عامل إيه"  # Add common words from your audio
        # result = stt.transcribe_file(audio_file, language="ar", initial_prompt=custom_prompt)
        
        # Print full transcription to console
        print("\n" + "="*60)
        print("FULL TRANSCRIPTION:")
        print("="*60)
        print(result['text'])
        
        # Save to file (without timestamps)
        stt.save_transcription(result, "transcription.txt")
        
        # Save with timestamps
        stt.save_transcription(result, "transcription_with_timestamps.txt", include_timestamps=True)
        
    else:
        print(f"Audio file '{audio_file}' not found!")
        print("\nTo use this script:")
        print("1. Place your Arabic audio file in the same directory")
        print("2. Update the 'audio_file' variable with your filename")
        print("3. Run the script again")
