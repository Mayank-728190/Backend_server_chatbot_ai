from flask import Flask, request, jsonify
from groq import Groq
import speech_recognition as sr
import os
import pyttsx3
import cv2  # for image recognition (OpenCV)
from PIL import Image
import numpy as np

app = Flask(__name__)

# Initialize Groq client
client = Groq(api_key="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")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Read the college data from the file
file_path = "collegedata.txt"
with open(file_path, 'r') as file:
    data = file.read()

memory = []

def generate_prompt(query, memory):
    memory_str = ' | '.join(memory) if len(memory) > 0 else "No questions were asked previously."
    prompt = (f"""
    Here is the provided data : {data}
    Answer the following query based on the information provided about colleges in Rajasthan...
    The given query may be from the context of the previous responses and the previous queries asked were : {memory_str}
    Here is the detailed query: {query}
    """)
    return prompt

@app.route('/query', methods=['POST'])
def handle_query():
    user_query = request.json.get('query')
    memory = request.json.get('memory')

    if not user_query:
        return jsonify({"error": "Query is required"}), 400
    if not memory:
        memory = []

    promp = generate_prompt(user_query, memory)

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": promp}],
            temperature=0.2,
            max_tokens=724,
            top_p=0.2,
            stream=True,
            stop=None,
        )

        response_text = ""
        for chunk in completion:
            response_text += chunk.choices[0].delta.content or ""

        # Text-to-Speech (Speak the response)
        engine.say(response_text)
        engine.runAndWait()

        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/voice-query', methods=['POST'])
def voice_query():
    text = recognize_speech_from_mic()
    return jsonify({"query": text})

def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."

@app.route('/image-recognition', methods=['POST'])
def image_recognition():
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({"error": "No image provided"}), 400

    # Load image for recognition
    image = Image.open(image_file.stream)
    image = np.array(image)

    # Simple image recognition logic (OpenCV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Image', gray)
    cv2.waitKey(1)  # Display image

    # Placeholder for further recognition logic
    return jsonify({"message": "Image processed successfully"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

