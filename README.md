# 🎧 Voice-Enabled Virtual Assistant for the Visually Impaired

This Python-based voice assistant is designed to assist **visually impaired users**—especially those who are blind—by enabling hands-free interaction through speech. It combines speech recognition, computer vision, and multiple APIs to deliver real-time information via voice feedback.

---

## 🧠 Features

- 🔊 **Voice Interaction**: Recognizes and responds to spoken commands using speech recognition and text-to-speech.
- 💵 **Currency Detection**: Identifies Indian currency notes using the Roboflow API and a webcam.
- 🖼️ **Object Recognition**: Identifies objects in a single camera snapshot using Imagga.
- 🚧 **Real-time Obstacle Warnings**: Uses a pretrained YOLO model to monitor camera frames and speak when a large detected object appears close and in front. Say “start obstacle detection”; press Q in the camera window to stop.
- 🌄 **On-demand Scene Description**: Say “describe scene” to capture one frame and hear a short sentence summarizing the main image tags.
- 🌦️ **Weather Updates**: Provides real-time temperature, humidity, and weather conditions.
- 📰 **News Headlines**: Fetches latest news based on your location via NewsAPI.
- 🌐 **Translation**: Converts English text into Hindi using Google Translate.
- 🕒 **Time Telling**: Announces the current time.
- 📻 **Voice Shutdown**: Say “shutdown” to safely exit the program.

---

## 🔧 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🛆 Dependencies

- `pyttsx3` – Offline text-to-speech
- `SpeechRecognition` – Voice input
- `googletrans==4.0.0-rc1` – Language translation
- `opencv-python` – Webcam capture
- `requests` – API communication
- `pyaudio` – Microphone access
- `ultralytics` – YOLO object detection for obstacle warnings (downloads the model weights on first run)

---

## 🔐 APIs Required

Make sure to register and insert your API keys:

| Service        | Purpose            | Link                                                     |
| -------------- | ------------------ | -------------------------------------------------------- |
| Roboflow       | Currency Detection | [https://roboflow.com](https://roboflow.com)             |
| Imagga         | Object Recognition | [https://imagga.com](https://imagga.com)                 |
| OpenWeatherMap | Weather Updates    | [https://openweathermap.org](https://openweathermap.org) |
| NewsAPI        | News Headlines     | [https://newsapi.org](https://newsapi.org)               |

---

## ♿ Accessibility Focus

This project is built with accessibility in mind, aiming to **assist blind and visually impaired individuals** in navigating daily tasks using only their voice and hearing. It provides a seamless, screen-free experience.

---

## 🛠️ Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Pranavgit2008/Blind-Beacon.git
   ```
2. Set the API keys as environment variables (see the API table below). For obstacle detection, install `ultralytics`; the YOLO weights download on first run.
3. Run the script:
   ```bash
   python main.py
   ```

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).



## 🚧 Obstacle detection safety note

Obstacle warnings are based on detected object size and position in the camera image; they do **not** measure real-world distance and can miss obstacles or produce false alerts. This experimental feature must not replace a white cane, guide dog, or other established mobility aid. Use it only as a supplementary aid and test it carefully in a safe, controlled environment.
