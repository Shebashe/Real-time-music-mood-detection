import cv2
import numpy as np
import pygame
import random
import os
import time
from tensorflow.keras.models import load_model


model = load_model(r"C:\Users\sheba\Downloads\music_model.h5")


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


music_paths = {
    'Angry':    r"C:\Users\sheba\Downloads\Montagem Coral Phonk Mp3 Ringtone Download - MobCup.Com.Co.mp3",
    'Disgust':  r"C:\Users\sheba\Downloads\Maatikkinaaru Orutharu.mp3",
    'Fear':     r"C:\Users\sheba\Downloads\Sad Violin (the Meme One) Sound Effects Download - MobCup.Com.Co.mp3",
    'Happy':    r"C:\Users\sheba\Downloads\serikalampa_s7a37452.mp3",
    'Neutral':  r"C:\Users\sheba\Downloads\yamma-yamma-vmusiq-com-mp3cut-net-21004.mp3",
    'Sad':      r"C:\Users\sheba\Downloads\Ini Kanneer Onnum Venda - Aattuthottil _ Athiran _ Malayalam.mp3",
    'Surprise': r"C:\Users\sheba\Downloads\Anime Wow Sound Effects Download - MobCup.Com.Co.mp3"
}

# Initialize Pygame mixer
pygame.mixer.init()


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def play_music(emotion):
    song_path = music_paths.get(emotion)
    if not song_path or not os.path.exists(song_path):
        print(f"No music file found for emotion: {emotion}")
        return
    print(f"Playing: {song_path}")
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()


def run_system():
    cap = cv2.VideoCapture(0)
    current_emotion = None
    last_change_time = 0
    cooldown = 5  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_color = frame[y:y+h, x:x+w]
            roi = cv2.resize(roi_color, (64, 64))  # Match model input
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=0)  # Shape: (1, 64, 64, 3)

            preds = model.predict(roi, verbose=0)
            emotion_idx = np.argmax(preds)
            emotion = emotion_labels[emotion_idx]
            confidence = np.max(preds)

            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Change music if emotion changed and cooldown passed
            if emotion != current_emotion and (time.time() - last_change_time) > cooldown:
                current_emotion = emotion
                last_change_time = time.time()
                pygame.mixer.music.stop()
                play_music(emotion)


        cv2.imshow('Real-Time Emotion Detection & Music Recommendation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    pygame.mixer.music.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_system()
