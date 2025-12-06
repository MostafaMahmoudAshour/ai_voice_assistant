import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

# For local LLM - we'll use Ollama 
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama not installed. Install with: pip install ollama")


class IntentType(Enum):
    """Types of user intents"""
    SEARCH_WEB = "search_web"
    OPEN_APPLICATION = "open_application"
    CLOSE_APPLICATION = "close_application"
    SYSTEM_CONTROL = "system_control"
    FILE_OPERATION = "file_operation"
    MEDIA_CONTROL = "media_control"
    INFORMATION_QUERY = "information_query"
    CONVERSATION = "conversation"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """Structured intent information"""
    intent_type: IntentType
    confidence: float
    entities: Dict[str, Any]
    original_text: str
    action: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ArabicNLU:
    """Natural Language Understanding for Arabic commands"""
    
    def __init__(self, model_name="llama3.1"):
        """
        Initialize NLU with local LLM
        
        Args:
            model_name: Ollama model name (e.g., "llama3.2:3b", "mistral", "gemma2:2b")
        """
        self.model_name = model_name
        
        if not OLLAMA_AVAILABLE:
            raise ImportError("Please install ollama: pip install ollama")
        
        # Check if Ollama is running
        try:
            ollama.list()
            print(f"✓ Connected to Ollama")
            print(f"✓ Using model: {model_name}")
        except Exception as e:
            print(f"❌ Error: Ollama is not running!")
            print("Please start Ollama first:")
            print("  1. Download from: https://ollama.com/download")
            print("  2. Install and run Ollama")
            print(f"  3. Pull the model: ollama pull {model_name}")
            raise
        
        # Intent patterns (for quick classification)
        self.intent_patterns = {
            IntentType.SEARCH_WEB: [
                "ابحث", "دور", "اعرف", "عايز اعرف", "ايه هو", "معلومات عن",
                "search", "find", "look for", "what is"
            ],
            IntentType.OPEN_APPLICATION: [
                "افتح", "شغل", "open", "start", "launch", "run"
            ],
            IntentType.CLOSE_APPLICATION: [
                "قفل", "اقفل", "close", "exit", "quit", "stop"
            ],
            IntentType.MEDIA_CONTROL: [
                "الصوت", "volume", "play", "pause", "stop", "next", "previous",
                "شغل", "وقف", "التالي", "السابق", "عالي", "واطي", "اعلى", "اخفض"
            ],
            IntentType.SYSTEM_CONTROL: [
                "shutdown", "restart", "sleep", "lock", "اقفل الجهاز", "اعد تشغيل"
            ],
            IntentType.FILE_OPERATION: [
                "احذف", "انسخ", "انقل", "delete", "copy", "move", "create", "save"
            ],
        }
    
    def create_nlu_prompt(self, text: str) -> str:
        """Create a prompt for the LLM to understand intent"""
        prompt = f"""You are an AI assistant that understands Arabic and English commands. Analyze the following user command and extract:

1. Intent Type (choose ONE):
   - search_web: User wants to search the internet
   - open_application: User wants to open a program/app
   - close_application: User wants to close a program/app
   - system_control: User wants to control system (shutdown, restart, etc.)
   - file_operation: User wants to work with files
   - media_control: User wants to control media (volume, play, pause, etc.)
   - information_query: User is asking for information
   - conversation: General conversation/greeting
   - unknown: Cannot determine intent

2. Entities: Extract key information (app names, search terms, file names, etc.)

3. Action: Specific action to take

4. Parameters: Any additional parameters needed

User Command: "{text}"

Respond ONLY with valid JSON in this exact format:
{{
    "intent": "intent_type_here",
    "confidence": 0.95,
    "entities": {{}},
    "action": "specific_action",
    "parameters": {{}}
}}

Examples:

User: "ابحث عن الطقس في القاهرة"
{{
    "intent": "search_web",
    "confidence": 0.95,
    "entities": {{"query": "الطقس في القاهرة", "location": "القاهرة"}},
    "action": "search",
    "parameters": {{"search_query": "weather in Cairo"}}
}}

User: "افتح جوجل كروم"
{{
    "intent": "open_application",
    "confidence": 0.98,
    "entities": {{"application": "google chrome"}},
    "action": "open",
    "parameters": {{"app_name": "chrome"}}
}}

User: "الصوت عالي"
{{
    "intent": "media_control",
    "confidence": 0.90,
    "entities": {{"control_type": "volume", "direction": "up"}},
    "action": "volume_up",
    "parameters": {{"amount": 10}}
}}

Now analyze the user command and respond with JSON only."""

        return prompt
    
    def understand(self, text: str) -> Intent:
        """
        Understand user intent from text
        
        Args:
            text: User's text command (from speech)
        
        Returns:
            Intent object with classification and entities
        """
        print(f"\n🧠 Understanding: '{text}'")
        
        # Quick pattern matching for common intents (faster)
        quick_intent = self._quick_intent_match(text)
        
        # Use LLM for detailed understanding
        try:
            prompt = self.create_nlu_prompt(text)
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.1,  # Low temperature for more deterministic output
                    'num_predict': 200,  # Limit response length
                }
            )
            
            # Extract JSON from response
            response_text = response['message']['content'].strip()
            
            # Remove markdown code blocks if present
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            # Parse JSON
            result = json.loads(response_text.strip())
            
            # Create Intent object
            intent = Intent(
                intent_type=IntentType(result['intent']),
                confidence=result.get('confidence', 0.5),
                entities=result.get('entities', {}),
                original_text=text,
                action=result.get('action'),
                parameters=result.get('parameters', {})
            )
            
            print(f"✓ Intent: {intent.intent_type.value}")
            print(f"✓ Confidence: {intent.confidence:.2f}")
            print(f"✓ Action: {intent.action}")
            if intent.entities:
                print(f"✓ Entities: {intent.entities}")
            
            return intent
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse LLM response: {e}")
            print(f"Response was: {response_text}")
            # Fallback to quick match
            return self._create_fallback_intent(text, quick_intent)
        
        except Exception as e:
            print(f"❌ Error in NLU: {e}")
            return self._create_fallback_intent(text, quick_intent)
    
    def _quick_intent_match(self, text: str) -> Optional[IntentType]:
        """Quick pattern matching for common intents"""
        text_lower = text.lower()
        
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return intent_type
        
        return None
    
    def _create_fallback_intent(self, text: str, quick_intent: Optional[IntentType]) -> Intent:
        """Create a fallback intent when LLM fails"""
        intent_type = quick_intent if quick_intent else IntentType.UNKNOWN
        
        return Intent(
            intent_type=intent_type,
            confidence=0.5,
            entities={},
            original_text=text,
            action=None,
            parameters={}
        )
    
    def extract_entities(self, text: str, intent_type: IntentType) -> Dict[str, Any]:
        """
        Extract specific entities based on intent type
        
        Args:
            text: User's text
            intent_type: Classified intent
        
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        text_lower = text.lower()
        
        if intent_type == IntentType.SEARCH_WEB:
            # Extract search query (remove command words)
            for keyword in ["ابحث عن", "دور على", "search for", "find"]:
                if keyword in text_lower:
                    entities['query'] = text_lower.split(keyword, 1)[1].strip()
                    break
            if 'query' not in entities:
                entities['query'] = text
        
        elif intent_type == IntentType.OPEN_APPLICATION:
            # Extract app name
            for keyword in ["افتح", "شغل", "open"]:
                if keyword in text_lower:
                    entities['application'] = text_lower.split(keyword, 1)[1].strip()
                    break
        
        elif intent_type == IntentType.MEDIA_CONTROL:
            # Detect volume control
            if any(word in text_lower for word in ["عالي", "اعلى", "up", "increase"]):
                entities['control_type'] = 'volume'
                entities['direction'] = 'up'
            elif any(word in text_lower for word in ["واطي", "اخفض", "down", "decrease"]):
                entities['control_type'] = 'volume'
                entities['direction'] = 'down'
            elif any(word in text_lower for word in ["شغل", "play"]):
                entities['control_type'] = 'playback'
                entities['action'] = 'play'
            elif any(word in text_lower for word in ["وقف", "pause"]):
                entities['control_type'] = 'playback'
                entities['action'] = 'pause'
        
        return entities


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("ARABIC NLU SYSTEM - Testing")
    print("="*60)
    
    # Initialize NLU
    nlu = ArabicNLU(model_name="llama3.1")  # Use smaller model for speed
    
    # Test commands
    test_commands = [
        "Search for weather in cairo",
        "Open Google Chrome",
        "Volume Up",
        "Play the Video",
        "Close the program",
        "What is AI",
        "Hello, How are you?",
    ]
    # test_commands = [
    #     "ابحث عن الطقس في القاهرة",
    #     "افتح جوجل كروم",
    #     "الصوت عالي",
    #     "شغل الفيديو",
    #     "قفل البرنامج",
    #     "ايه هو الذكاء الاصطناعي",
    #     "إزيك عامل إيه",
    # ]
    
    print("\nTesting NLU with sample commands:")
    print("="*60)
    
    for command in test_commands:
        intent = nlu.understand(command)
        print("\n" + "-"*60)
        print(f"Command: {command}")
        print(f"Intent: {intent.intent_type.value}")
        print(f"Confidence: {intent.confidence:.2f}")
        print(f"Action: {intent.action}")
        print(f"Entities: {json.dumps(intent.entities, ensure_ascii=False, indent=2)}")
        print("-"*60)
    
    print("\n✓ NLU Testing Complete!")