# Hand Sign to Text Translator

Recognizes hand gestures through your webcam and turns them into text on screen, in real time.

## How it works

Uses OpenCV to grab webcam frames and MediaPipe's HandLandmarker (Tasks API) to find 21 points on the hand. From there I check which fingers are up/down and match that against a few known gestures. When one matches, it shows up as text and you can add it to a sentence.

## Gestures it knows right now

- Thumbs up = YES
- Index + middle up = PEACE
- Fist = STOP
- All fingers up = HELLO
- Just index finger = POINT
- Thumb + pinky = CALL ME

## Controls

- SPACE - add detected word to sentence
- BACKSPACE - undo last word
- c - clear sentence
- q - quit

## Running it

```
pip install -r requirements.txt
python sign_to_text.py
```

First run downloads the hand landmark model automatically

## Notes

Single hand for now, two-hand support coming. Gesture matching is just finger position rules, not a trained model, so lighting/angle matters significantly.
