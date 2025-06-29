"""
This voice-enabled assistant is designed to help visually impaired users perform tasks through speech. 
It can recognize spoken commands, detect Indian currency using a webcam, identify nearby objects, and provide weather and news updates. 
The assistant also supports English-to-Hindi translation and announces the current time. 
Designed for accessibility, it offers hands-free interaction and audio feedback to assist blind users. 
This code relies on multiple APIs and modules for optimal functionality. 

You must obtain API keys from these websites:

1. Roboflow API - For detecting Indian currency notes via webcam images.
🔗 https://roboflow.com
2. Imagga API - For object detection and tagging from webcam images.
🔗 https://imagga.com
3. OpenWeatherMap API - To fetch live weather information.
🔗 https://openweathermap.org
4. NewsAPI - To fetch the latest news headlines based on the city.
🔗 https://newsapi.org

Please ensure all required dependencies are properly installed and configured.

"""

import pyttsx3 
import speech_recognition as sr 
import datetime
import cv2
import requests

# voice selecting function
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# print(voices[1].id)
engine.setProperty('voice', voices[1].id)
rate = engine.getProperty('rate')   # Default rate is usually around 200
engine.setProperty('rate', 180)

#speak function
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

#Wishing function
def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("Good Morning!")

    elif hour>=12 and hour<18:
        speak("Good Afternoon!")   

    else:
        speak("Good Evening!")         

#It takes microphone input from the user and returns string output
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        print("Recognizing...")    
        query = r.recognize_google(audio, language='en-in')
        print(f"User said:{query}")

    except Exception as e:
        # print(e)    
        print("Say that again please...")  
        return "None"
    return query
wishMe()

# weather info gathering function using api of openweather
def get_weather(api_key, city, country_code=''):   
    base_url = "http://api.openweathermap.org/data/2.5/weather"
                
    params = {
        'q': f'{city},{country_code}',
        'appid': api_key,
            }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        weather_data = response.json()
        return weather_data
    else:
        return None

# Main function
while True:
    query = takeCommand().lower()
    # Translator
    from googletrans import Translator
    translator=Translator()
    def Translate(translate):
            # Text to translate
            text_to_translate =translate
            # Translate from English to Hindi
            translation = translator.translate(text_to_translate, src='en', dest='hi')
            translated_text = translation.text
            # print("Original Text:", text_to_translate)
            print("Translated Text:", translated_text)
            speak(translated_text)

    # Currency Detection using API
    if 'currency' in query:
            # Defineing API key 
            ROBOFLOW_API_KEY = "Replace with your ROBOFLOW API" #API key is available at ROBOFLOW
            MODEL_VERSION = "1"
            PROJECT_NAME = "indian-currency-detector-boyiq"

            # endpoint 
            url = f"https://detect.roboflow.com/{PROJECT_NAME}/{MODEL_VERSION}"
            params = {
                "api_key": ROBOFLOW_API_KEY,
                "confidence": 0.8,  # Adjust as needed
                "overlap": 0.5
            }

            # Selecting the webcam feed
            cap = cv2.VideoCapture(0) #Change camera input

            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()

                if not ret:
                    print("Failed to grab frame")
                    break

                
                _, img_encoded = cv2.imencode('.jpg', frame)
                image_data = img_encoded.tobytes()

                # Posting the image to Roboflow API
                response = requests.post(url, params=params, files={"file": image_data})

                predictions = response.json()

                # Drawing boxes on the frame
                if response.status_code == 200:
                    for prediction in predictions['predictions']:
                        x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
                        label = prediction['class']

                        x1, y1 = int(x - width / 2), int(y - height / 2)
                        x2, y2 = int(x + width / 2), int(y + height / 2)

                        # Drawing bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        Str1=label.split("-")[0]
                        Str2=label.split("-")[1]
                        Str=Str1+'_'+Str2
                        # print(Str,"Rupees")
                        if Str=='five_hundred':
                            # print("500 Rupees")
                            var="500 Rupees"
                            speak(var) 

                        elif Str=='two_thousand':
                            print("2000 Rupees")
                            var="2000 Rupees"
                            speak(var)
                        
                        elif Str=='ten_front':
                            print("10 Rupees")
                            var="10 Rupees"
                            speak(var)

                        elif Str=='ten_back':
                            print("10 Rupees")
                            var="10 Rupees"
                            speak(var)

                        elif Str=='hundred_front':
                            print("100 Rupees")
                            var="100 Rupees"
                            speak(var)
                        
                        elif Str=='hundred_back':
                            print("100 Rupees")
                            var="100 Rupees"
                            speak(var)

                        elif Str=='twenty_front':
                            print("20 Rupees")
                            var="20 Rupees"
                            speak(var)

                        elif Str=='twenty_back':
                            print("20 Rupees")
                            var="20 Rupees"
                            speak(var)

                        elif Str=='two_hundred':
                            print("200 Rupees")
                            var="200 Rupees"
                            speak(var)

                        elif Str=='fifty_front':
                            print("50 Rupees")
                            var="50 Rupees"
                            speak(var)
                        
                        elif Str=='fifty_back':
                            print("50 Rupees")
                            var="50 Rupees"
                            speak(var)


                # Displaying the frame with result
                cv2.imshow("Currency Detection", frame)

                # Exit on pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the capture and close windows
            cap.release()
            cv2.destroyAllWindows()

    # Image Detection using Imangga API
    elif 'detect' in query:
        import time
        import os

        # Imagga API key
        API_KEY = "Replace with your API key" #API key is available at Imangga                               
        API_SECRET = "Replace with your API secret key" #API secret key is available at Imangga
        auth = (API_KEY, API_SECRET)

        # Capture image from webcam
        def capture_image_from_webcam():
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                image_path = "captured_image.jpg"
                cv2.imwrite(image_path, frame)
            cap.release()
            cv2.destroyAllWindows()
            return image_path

        # Posting the captured image to Imagga API
        def tag_image_file(image_path, confidence_threshold=30):
            url = "https://api.imagga.com/v2/tags"
            with open(image_path, "rb") as image_file:
                files = {"image": image_file}
                response = requests.post(url, auth=auth, files=files)
            
            if response.status_code == 200:
                tags = response.json().get("result", {}).get("tags", [])
                filtered_tags = [tag["tag"]["en"] for tag in tags if tag["confidence"] > confidence_threshold]
                return filtered_tags
            else:
                print("Error:", response.json())
                return None

        
        image_path = capture_image_from_webcam()
        tags = tag_image_file(image_path)

        if tags:
            var1="I found " + ", ".join(tags) + " around you."
            print(var1)
            speak("I found " + ", ".join(tags) + " around you.")
        else:
            print("No high-confidence tags found.")
        time.sleep(4)
        if os.path.exists(image_path):
            os.remove(image_path)
        else:
            print("Image file not found.")

    # Time
    elif 'the time' in query:
        strTime = datetime.datetime.now().strftime("%I:%M %p")
        strTime=strTime.replace(":","")    
        speak(f"The time is {strTime}")

    # Weather
    elif 'the weather' in query:
        def display_weather(weather_data):
            if weather_data:
                city_name="Replace with your city"
                main_info = weather_data['main']
                temperature_kelvin = main_info['temp']
                humidity = main_info['humidity']
                description = weather_data['weather'][0]['description']
                # Converting kelvin to Celsius
                temperature_celsius = temperature_kelvin - 273.15

                print(f'Temperature in {city_name} is {temperature_celsius:.0f}°C and Humidity is {humidity}%')
                speak(f'Temperature in {city_name} is {temperature_celsius:.0f}°C and Humidity is {humidity}%')
                print(f'Feels like {description}')
                speak(f'Feels like {description}')
            else:
                print('Failed to fetch weather data.')
                speak('Failed to fetch weather data.')
        if __name__ == "__main__":
            api_key = 'Replace it with your API key' #API key is available at Openweathermap
            city_name = 'Replace it with your city'
            weather_data = get_weather(api_key, city_name)

            display_weather(weather_data)

    # News
    elif 'news' in query:
        def fetch_news_headlines(api_key, num_headlines=10):
            # News API endpoint
            url = f"https://newsapi.org/v2/everything?q={city_name}&language=en&apiKey={api_key}&pageSize={num_headlines}"
            
            # Make a request to the News API
            response = requests.get(url)
            
            # Check if the request was successful
            if response.status_code == 200:
                news_data = response.json()
                return news_data['articles']  
            else:
                print("Error:", response.status_code)
                return []

        def print_headlines(articles):
            for article in articles:
                print(f"- {article['title']}")
                str=f"- {article['title']}"
                speak(str)

        if __name__ == "__main__":
            api_key = "Replace it with your API key"  # API key available at newsapi
            headlines = fetch_news_headlines(api_key,)

            if headlines:
                print(f"Latest News Headlines related to {city_name}:")
                speak(f"Latest News Headlines related to {city_name}:")
                print_headlines(headlines)
                
            else:
                print("No headlines found.")
                speak("No headlines found.")

    # Translator works with hindi
    elif 'translate' in query:
        engine.setProperty('voice', voices[0].id) #Change voice here
        rate = engine.getProperty('rate')   
        engine.setProperty('rate', 180) # Default rate is usually around 200

        try:
            speak("Translating...")
            translate = query.replace("translate","")
            translate = translate.replace("to hindi","")
            # print(translate)
            Translate(translate)
        except Exception as e:
            speak("I'am unable to translate the text to hindi")

    # To Close the program
    elif 'shutdown' in query:
        print("Shuting Down...")
        speak("Shuting Down")
        exit()

