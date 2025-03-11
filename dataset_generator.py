from pathlib import Path

def generate_training_data(num_entries: int = 1000, output_file: str = "train_data.yaml") -> None:
    """
    Generate a large-scale training dataset for AuroraModel and save it to a YAML file.
    
    Args:
        num_entries (int): Number of data entries to generate (default: 1000)
        output_file (str): Output YAML file path (default: "train_data.yaml")
    """
    output = []
    
    # First 20 manual entries for variety
    manual_entries = [
        {"text": "Hello, how can I assist you today?", "response": 1, "audio_file": "sample_0001.wav"},
        {"text": "I need help with my account.", "response": 0, "audio_file": "sample_0002.wav"},
        {"text": "Can you tell me the weather forecast?", "response": 1, "audio_file": "sample_0003.wav"},
        {"text": "Goodbye, I don’t need assistance.", "response": 0, "audio_file": "sample_0004.wav"},
        {"text": "What’s the status of my order?", "response": 1, "audio_file": "sample_0005.wav"},
        {"text": "This is urgent, please help!", "response": 0, "audio_file": "sample_0006.wav"},
        {"text": "Tell me about your services.", "response": 1, "audio_file": "sample_0007.wav"},
        {"text": "I’m not interested, thanks.", "response": 0, "audio_file": "sample_0008.wav"},
        {"text": "How do I reset my password?", "response": 1, "audio_file": "sample_0009.wav"},
        {"text": "Your system is too slow.", "response": 0, "audio_file": "sample_0010.wav"},
        {"text": "Can you schedule an appointment?", "response": 1, "audio_file": "sample_0011.wav"},
        {"text": "I want to speak to a human.", "response": 0, "audio_file": "sample_0012.wav"},
        {"text": "What are your operating hours?", "response": 1, "audio_file": "sample_0013.wav"},
        {"text": "This isn’t working properly.", "response": 0, "audio_file": "sample_0014.wav"},
        {"text": "Please provide pricing details.", "response": 1, "audio_file": "sample_0015.wav"},
        {"text": "I’m frustrated with this service.", "response": 0, "audio_file": "sample_0016.wav"},
        {"text": "Can you send me a confirmation?", "response": 1, "audio_file": "sample_0017.wav"},
        {"text": "No thanks, I’ll figure it out.", "response": 0, "audio_file": "sample_0018.wav"},
        {"text": "What’s the best plan for me?", "response": 1, "audio_file": "sample_0019.wav"},
        {"text": "Your support is terrible.", "response": 0, "audio_file": "sample_0020.wav"},
    ]
    output.extend(manual_entries)
    
    # Generate remaining entries
    for i in range(21, num_entries + 1):
        text_parts = []
        if i % 2 == 0:
            text_parts.append(f"Request {i // 2}: ")
        if i % 3 == 0:
            text_parts.append("Please ")
        if i % 5 == 0:
            text_parts.append("Quickly ")
        if i % 7 == 0:
            text_parts.append("Urgently ")
        if i % 2 == 0:
            text_parts.append("help me with ")
        if i % 3 == 1:
            text_parts.append("tell me about ")
        
        suffix = i % 4
        if suffix == 0:
            text_parts.append("my account")
        elif suffix == 1:
            text_parts.append("the system")
        elif suffix == 2:
            text_parts.append("your services")
        else:
            text_parts.append("something else")
        
        text = "".join(text_parts).strip()
        response = 1 if i % 2 == 0 else 0
        audio_file = f"sample_{i:04d}.wav"
        
        output.append({"text": text, "response": response, "audio_file": audio_file})
    
    # Write to file
    with open(output_file, "w") as f:
        for entry in output:
            f.write(f"- text: \"{entry['text']}\"\n")
            f.write(f"  response: {entry['response']}\n")
            f.write(f"  audio_file: \"{entry['audio_file']}\"\n")

if __name__ == "__main__":
    generate_training_data()