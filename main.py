"""Voice-enabled assistant for visually impaired users.

Currency and image-tagging services require API credentials supplied through
environment variables. This application uses cloud services; it is not a
replacement for a mobility aid or professional navigation system.
"""

import datetime
import os
import re
import time

import cv2
import pyttsx3
import requests
import speech_recognition as sr
from googletrans import Translator

# ----------------------------- Configuration -----------------------------
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_PROJECT = os.getenv(
    "ROBOFLOW_PROJECT", "indian-currency-detector-boyiq"
)
ROBOFLOW_VERSION = os.getenv("ROBOFLOW_VERSION", "1")
IMAGGA_API_KEY = os.getenv("IMAGGA_API_KEY", "")
IMAGGA_API_SECRET = os.getenv("IMAGGA_API_SECRET", "")
REQUEST_TIMEOUT = 15
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# ------------------------------ Voice output -----------------------------
engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")
if voices:
    # Prefer the second installed voice when available; otherwise use the first.
    engine.setProperty("voice", voices[1 if len(voices) > 1 else 0].id)
engine.setProperty("rate", 180)


def speak(message):
    """Print and speak a message."""
    print(message)
    engine.say(str(message))
    engine.runAndWait()


def wish_me():
    hour = datetime.datetime.now().hour
    if hour < 12:
        speak("Good morning!")
    elif hour < 18:
        speak("Good afternoon!")
    else:
        speak("Good evening!")


def take_command():
    """Listen for an Indian-English voice command; return an empty string on failure."""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.pause_threshold = 1
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=10)
        query = recognizer.recognize_google(audio, language="en-in")
        print(f"You said: {query}")
        return query.lower().strip()
    except sr.WaitTimeoutError:
        print("No speech detected.")
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand. Please try again.")
    except sr.RequestError as exc:
        print(f"Speech recognition service error: {exc}")
        speak("Speech recognition is currently unavailable.")
    except (OSError, AttributeError) as exc:
        print(f"Microphone error: {exc}")
        speak("I cannot access the microphone. Please check its connection.")
    return ""


# ---------------------------- Currency detection -------------------------
# Supports common model labels such as "five-hundred", "five_hundred",
# "hundred_front", and "hundred_back". Update this map if your model uses
# different class names.
CURRENCY_LABELS = {
    "five hundred": "500 rupees",
    "two thousand": "2000 rupees",
    "ten front": "10 rupees",
    "ten back": "10 rupees",
    "twenty front": "20 rupees",
    "twenty back": "20 rupees",
    "fifty front": "50 rupees",
    "fifty back": "50 rupees",
    "hundred front": "100 rupees",
    "hundred back": "100 rupees",
    "two hundred": "200 rupees",
    # Alternate label orders sometimes used by datasets:
    "front ten": "10 rupees",
    "back ten": "10 rupees",
    "front twenty": "20 rupees",
    "back twenty": "20 rupees",
    "front fifty": "50 rupees",
    "back fifty": "50 rupees",
    "front hundred": "100 rupees",
    "back hundred": "100 rupees",
}


def normalize_label(label):
    """Normalize punctuation and separators so label matching is resilient."""
    return re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()


def denomination_for_label(label):
    normalized = normalize_label(label)
    if normalized in CURRENCY_LABELS:
        return CURRENCY_LABELS[normalized]

    # Match denomination names even when the model appends a side/version tag.
    tokens = normalized.split()
    if "thousand" in tokens and "two" in tokens:
        return "2000 rupees"
    if "hundred" in tokens and "five" in tokens:
        return "500 rupees"
    if "hundred" in tokens and "two" in tokens:
        return "200 rupees"
    if "hundred" in tokens:
        return "100 rupees"
    if "fifty" in tokens:
        return "50 rupees"
    if "twenty" in tokens:
        return "20 rupees"
    if "ten" in tokens:
        return "10 rupees"
    return None


def detect_currency():
    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY.lower().startswith("replace"):
        speak("Currency detection is not configured. Set the ROBOFLOW_API_KEY environment variable.")
        return

    url = f"https://detect.roboflow.com/{ROBOFLOW_PROJECT}/{ROBOFLOW_VERSION}"
    params = {"api_key": ROBOFLOW_API_KEY, "confidence": 0.65, "overlap": 0.5}
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        speak("I could not open the camera. Please check the camera connection.")
        cap.release()
        return

    last_spoken = {}
    speak("Currency detection started. Press Q in the camera window to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                speak("I could not capture an image from the camera.")
                break

            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                print("Could not encode camera frame.")
                continue

            try:
                response = requests.post(
                    url,
                    params=params,
                    files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                payload = response.json()
                predictions = payload.get("predictions", [])
            except (requests.RequestException, ValueError) as exc:
                print(f"Roboflow request failed: {exc}")
                cv2.imshow("Blind-Beacon - Currency Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.2)
                continue

            now = time.monotonic()
            for prediction in predictions:
                label = prediction.get("class", "unknown")
                try:
                    confidence = float(prediction.get("confidence", 0))
                    x = float(prediction["x"])
                    y = float(prediction["y"])
                    width = float(prediction["width"])
                    height = float(prediction["height"])
                except (KeyError, TypeError, ValueError):
                    continue

                x1 = max(0, int(x - width / 2))
                y1 = max(0, int(y - height / 2))
                x2 = min(frame.shape[1] - 1, int(x + width / 2))
                y2 = min(frame.shape[0] - 1, int(y + height / 2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(
                    frame, f"{label} {confidence:.0%}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2
                )

                denomination = denomination_for_label(label)
                # Speak only sufficiently confident detections and throttle repeats.
                if denomination and confidence >= 0.65:
                    if now - last_spoken.get(denomination, -30) >= 3:
                        speak(denomination)
                        last_spoken[denomination] = now

            cv2.imshow("Blind-Beacon - Currency Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        speak("Currency detection stopped.")


# ------------------------------ Object tagging ---------------------------
def capture_image_from_webcam():
    """Capture one frame and return its image path, or None on failure."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    try:
        if not cap.isOpened():
            speak("I could not open the camera.")
            return None
        ok, frame = cap.read()
        if not ok:
            speak("I could not capture an image.")
            return None
        image_path = "captured_image.jpg"
        if not cv2.imwrite(image_path, frame):
            speak("I could not save the captured image.")
            return None
        return image_path
    finally:
        cap.release()
        cv2.destroyAllWindows()


def tag_image_file(image_path, confidence_threshold=30):
    """Upload an image to Imagga and return high-confidence English tags."""
    if not IMAGGA_API_KEY or not IMAGGA_API_SECRET:
        speak("Object tagging is not configured. Set the IMAGGA_API_KEY and IMAGGA_API_SECRET environment variables.")
        return None

    url = "https://api.imagga.com/v2/tags"
    try:
        with open(image_path, "rb") as image_file:
            response = requests.post(
                url,
                auth=(IMAGGA_API_KEY, IMAGGA_API_SECRET),
                files={"image": image_file},
                timeout=REQUEST_TIMEOUT,
            )
        response.raise_for_status()
        data = response.json()
        tags = data.get("result", {}).get("tags", [])
        return [
            item.get("tag", {}).get("en", "")
            for item in tags
            if float(item.get("confidence", 0)) >= confidence_threshold
            and item.get("tag", {}).get("en")
        ]
    except (OSError, requests.RequestException, ValueError, TypeError) as exc:
        print(f"Imagga request failed: {exc}")
        speak("I was unable to analyze the image.")
        return None


def detect_objects():
    image_path = capture_image_from_webcam()
    if not image_path:
        return
    try:
        tags = tag_image_file(image_path)
        if tags:
            # Limit the spoken list so the user is not overwhelmed by generic tags.
            message = "I found: " + ", ".join(tags[:5]) + "."
            speak(message)
        elif tags == []:
            speak("I could not identify any objects with sufficient confidence.")
    finally:
        try:
            os.remove(image_path)
        except OSError:
            pass



# ------------------------ Real-time obstacle detection ------------------
def detect_obstacles():
    """Monitor the camera and announce likely nearby obstacles.
 
    This uses a pretrained COCO object detector. Bounding-box size and image
    position are only rough visual cues, not reliable physical distance.
    Press Q in the camera window to stop monitoring.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        speak("Obstacle detection needs the ultralytics package. Install requirements and try again.")
        return

    speak("Starting obstacle detection. Keep using your cane or guide dog. Press Q in the camera window to stop.")
    try:
        model = YOLO(os.getenv("OBSTACLE_MODEL", "yolov8n.pt"))
    except Exception as exc:
        print(f"Could not load obstacle detection model: {exc}")
        speak("I could not load the obstacle detection model.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        speak("I could not open the camera for obstacle detection.")
        return

    last_alert = {}
    alert_cooldown = 4.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                speak("Camera frame unavailable. Stopping obstacle detection.")
                break
            try:
                result = model.predict(frame, imgsz=640, conf=0.45, verbose=False)[0]
            except Exception as exc:
                print(f"Obstacle inference failed: {exc}")
                speak("Obstacle detection encountered an error.")
                break

            now = time.monotonic()
            frame_area = max(1, frame.shape[0] * frame.shape[1])
            if result.boxes is not None:
                for box in result.boxes:
                    try:
                        cls_id = int(box.cls[0].item())
                        label = str(model.names[cls_id]).lower()
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    except (ValueError, TypeError, IndexError, AttributeError):
                        continue
                    box_area = max(0, x2 - x1) * max(0, y2 - y1)
                    area_ratio = box_area / frame_area
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    # Warn on large objects near the center/lower part of the image.
                    centered = abs(center_x - frame.shape[1] / 2) < frame.shape[1] * 0.30
                    lower_or_center = center_y > frame.shape[0] * 0.40
                    if area_ratio >= 0.18 and centered and lower_or_center:
                        if now - last_alert.get(label, -100.0) >= alert_cooldown:
                            speak(f"Warning. {label} appears close ahead. Please proceed carefully.")
                            last_alert[label] = now
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"{label} {area_ratio:.0%}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            cv2.imshow("Blind-Beacon - Obstacle Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        speak("Obstacle detection stopped.")


# ------------------------------ Scene description -----------------------
def describe_scene():
    """Capture one frame and speak a short description based on image tags."""
    image_path = capture_image_from_webcam()
    if not image_path:
        return
    try:
        tags = tag_image_file(image_path, confidence_threshold=35)
        if tags:
            # Remove duplicate tags and keep the spoken description concise.
            unique_tags = list(dict.fromkeys(tag.lower() for tag in tags if tag))[:3]
            if len(unique_tags) == 1:
                speak(f"This scene appears to show {unique_tags[0]}.")
            else:
                speak("This scene appears to show " + ", ".join(unique_tags[:-1]) + " and " + unique_tags[-1] + ".")
        elif tags == []:
            speak("I could not identify the scene clearly.")
    finally:
        try:
            os.remove(image_path)
        except OSError:
            pass

# -------------------------------- Weather --------------------------------
def get_weather(api_key, city, country_code=""):
    if not api_key or api_key.lower().startswith("replace") or not city:
        speak("Weather is not configured. Set OPENWEATHER_API_KEY and WEATHER_CITY.")
        return None
    url = "https://api.openweathermap.org/data/2.5/weather"
    try:
        response = requests.get(
            url,
            params={"q": f"{city},{country_code}".strip(","), "appid": api_key},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"Weather request failed: {exc}")
        return None


def announce_weather():
    city = os.getenv("WEATHER_CITY", "")
    data = get_weather(os.getenv("OPENWEATHER_API_KEY", ""), city)
    if not data:
        speak("Failed to fetch weather information.")
        return
    main = data.get("main", {})
    weather = data.get("weather", [{}])[0]
    temp = main.get("temp")
    humidity = main.get("humidity")
    description = weather.get("description", "unknown conditions")
    if temp is None:
        speak("The weather service returned incomplete information.")
        return
    speak(
        f"Weather in {data.get('name', city)}: {temp - 273.15:.0f} degrees Celsius, "
        f"humidity {humidity if humidity is not None else 'unknown'} percent, "
        f"{description}."
    )


# --------------------------------- News ----------------------------------
def announce_news():
    api_key = os.getenv("NEWS_API_KEY", "")
    city = os.getenv("WEATHER_CITY", "")
    if not api_key or not city:
        speak("News is not configured. Set NEWS_API_KEY and WEATHER_CITY.")
        return
    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": city, "language": "en", "apiKey": api_key,
                "pageSize": 5, "sortBy": "publishedAt",
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        articles = response.json().get("articles", [])
        if not articles:
            speak("No news headlines were found.")
            return
        speak(f"Here are the latest headlines related to {city}.")
        for article in articles:
            title = article.get("title")
            if title:
                speak(title)
    except (requests.RequestException, ValueError) as exc:
        print(f"News request failed: {exc}")
        speak("I was unable to fetch the news.")


# ------------------------------- Translation -----------------------------
def translate_to_hindi(text):
    text = re.sub(r"^translate\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*to hindi\s*$", "", text, flags=re.IGNORECASE).strip()
    if not text:
        speak("Please say the English text you want translated, for example: translate hello to Hindi.")
        return
    try:
        translated = Translator().translate(text, src="en", dest="hi").text
        speak(translated)
    except Exception as exc:
        print(f"Translation failed: {exc}")
        speak("I was unable to translate that text.")


def main():
    wish_me()
    speak("Blind Beacon is ready. Say start obstacle detection, describe scene, currency, detect objects, the time, the weather, news, translate, or shutdown.")
    while True:
        query = take_command()
        if not query:
            continue

        if "shutdown" in query or "exit assistant" in query or "quit assistant" in query:
            speak("Shutting down Blind Beacon.")
            break
        elif "start obstacle detection" in query or "detect obstacles" in query or "obstacle detection" in query:
            detect_obstacles()
        elif "describe scene" in query or "describe the scene" in query or "describe surroundings" in query or "what does the scene look like" in query:
            describe_scene()
        elif "currency" in query:
            detect_currency()
        elif "detect" in query or "identify objects" in query or "what is around me" in query:
            detect_objects()
        elif "time" in query:
            speak(f"The time is {datetime.datetime.now().strftime('%I:%M %p')}.")
        elif "weather" in query:
            announce_weather()
        elif "news" in query or "headlines" in query:
            announce_news()
        elif "translate" in query:
            translate_to_hindi(query)
        else:
            speak("Sorry, I don't recognize that command. Please try again.")


if __name__ == "__main__":
    main()
