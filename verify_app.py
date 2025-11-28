import requests
import os

# URL of the running FastAPI server
API_URL = "http://127.0.0.1:8000/summarize"

# Path to a sample audio file (User needs to provide this)
AUDIO_FILE_PATH = "sample_audio.wav" 

def test_api():
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"Error: Please place a sample audio file at '{AUDIO_FILE_PATH}' to test.")
        return

    print(f"Sending {AUDIO_FILE_PATH} to {API_URL}...")
    
    with open(AUDIO_FILE_PATH, "rb") as f:
        files = {"file": f}
        try:
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("\n--- Success! API Response ---")
                print(f"Detected Language: {data.get('language')}")
                print(f"Summary: {data.get('summary')}")
                print(f"Action Items: {data.get('action_items')}")
                print("-----------------------------")
            else:
                print(f"Error: API returned status code {response.status_code}")
                print(response.text)
                
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to the server. Is it running? (uvicorn app.backend.main:app --reload)")

if __name__ == "__main__":
    test_api()
