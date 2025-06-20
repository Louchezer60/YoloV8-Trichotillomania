import cv2
import time
import random
import threading
import os
from gtts import gTTS
from playsound import playsound
from ultralytics import YOLO
import numpy as np

# Configuration
MODEL_PATH = r"C:\Users\yvan_\OneDrive\Documents\Trich\weights\trich_model_v2\weights\best.onnx"

# Load model
model = YOLO(MODEL_PATH, task='detect')
print("Model loaded - Using raw model")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Text-to-Speech Setup ---
encouraging_messages = [
    "You're doing great! Keep it up.",
    "Stay strong, you can overcome this.",
    "Remember your progress, don't give up.",
    "Take a deep breath and relax.",
    "You've got this!",
    "One step at a time, you can do this.",
    "Believe in yourself.",
    "Focus on your strength.",
    "You are stronger than you think."
]

# UI Colors
COLOR_NORMAL = (50, 205, 50)         # Green for normal state
COLOR_WARNING = (0, 165, 255)        # Orange for warning state
COLOR_ALERT = (0, 0, 255)            # Red for pulling detected
COLOR_INFO = (255, 255, 255)         # White for information
COLOR_BG = (40, 40, 40)              # Dark background
COLOR_PROGRESS = (0, 200, 255)       # Cyan for progress bars

# Variables for cooldown mechanism for speech
last_speech_time = 0
COOLDOWN_DURATION = 2  # Minimum 5 seconds cooldown between speech messages

# Variables for pulling detection duration
pulling_start_time = None
PULLING_DETECTION_THRESHOLD_DURATION = 0.5

# Minimum confidence for a "pulling" detection
MIN_PULLING_CONFIDENCE = 0.3

def speak_message(message):
    """Speaks the given message using gTTS and playsound."""
    filename = f"temp_audio_{int(time.time() * 1000)}.mp3"
    try:
        print(f"TTS: Generating audio for: '{message}'")
        tts = gTTS(text=message, lang='en')
        tts.save(filename)
        print(f"TTS: Audio saved to {filename}")
        
        print("TTS: Playing audio...")
        playsound(filename)
        print("TTS: Audio playback finished.")
        
        os.remove(filename)
        print(f"TTS: Temporary file {filename} removed.")
    except Exception as e:
        print(f"Error during gTTS or playsound for message '{message}': {e}")
        print("Please ensure you have an active internet connection for gTTS and a compatible audio player for .mp3 files.")
    finally:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"TTS: Cleaned up remaining file {filename}.")
            except Exception as e:
                print(f"TTS: Error removing file in finally block: {e}")

def draw_progress_bar(img, progress, position, size, color):
    """Draw a progress bar on the image."""
    x, y = position
    width, height = size
    # Draw background
    cv2.rectangle(img, (x, y), (x + width, y + height), (60, 60, 60), -1)
    # Draw progress
    fill_width = int(width * progress)
    cv2.rectangle(img, (x, y), (x + fill_width, y + height), color, -1)
    # Draw border
    cv2.rectangle(img, (x, y), (x + width, y + height), (200, 200, 200), 1)
    
def draw_status_box(img, text, color, position, size):
    """Draw a status box with text."""
    x, y = position
    width, height = size
    # Draw background
    cv2.rectangle(img, (x, y), (x + width, y + height), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (200, 200, 200), 2)
    
    # Put text
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
    text_x = x + (width - text_width) // 2
    text_y = y + (height + text_height) // 2
    cv2.putText(img, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

def draw_ui_header(img, status):
    """Draw the UI header with title and status."""
    # Draw header background
    cv2.rectangle(img, (0, 0), (img.shape[1], 70), COLOR_BG, -1)
    
    # Draw title
    cv2.putText(img, "TRICHOTILLOMANIA MONITOR", (20, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (220, 220, 220), 2)
    
    # Draw status
    status_text = "NORMAL"
    status_color = COLOR_NORMAL
    if status == "pulling":
        status_text = "PULLING DETECTED"
        status_color = COLOR_ALERT
    elif status == "warning":
        status_text = "WARNING"
        status_color = COLOR_WARNING
        
    cv2.putText(img, f"Status: {status_text}", (img.shape[1] - 300, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, status_color, 2)

def draw_ui_footer(img, cooldown_remaining):
    """Draw the UI footer with instructions and cooldown indicator."""
    # Draw footer background
    cv2.rectangle(img, (0, img.shape[0] - 70), (img.shape[1], img.shape[0]), COLOR_BG, -1)
    
    # Draw instructions
    cv2.putText(img, "Press 'Q' to exit", (20, img.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    
    # Draw cooldown status
    cooldown_text = f"Next encouragement: Ready"
    cooldown_color = COLOR_NORMAL
    
    if cooldown_remaining > 0:
        cooldown_text = f"Next encouragement in: {cooldown_remaining:.1f}s"
        cooldown_color = COLOR_WARNING
        
    cv2.putText(img, cooldown_text, (img.shape[1] - 400, img.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cooldown_color, 2)

def draw_info_panel(img, fps, confidence, class_label):
    """Draw the information panel on the right side."""
    # Draw panel background
    panel_width = 300
    cv2.rectangle(img, (img.shape[1] - panel_width, 70), 
                 (img.shape[1], img.shape[0] - 70), 
                 (50, 50, 50), -1)
    cv2.rectangle(img, (img.shape[1] - panel_width, 70), 
                 (img.shape[1], img.shape[0] - 70), 
                 (100, 100, 100), 2)
    
    # Draw title
    cv2.putText(img, "SYSTEM INFORMATION", 
               (img.shape[1] - panel_width + 20, 110), 
               cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 255), 2)
    
    # Draw FPS
    cv2.putText(img, f"FPS: {int(fps)}", 
               (img.shape[1] - panel_width + 20, 160), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_INFO, 2)
    
    # Draw detection info
    if confidence > 0:
        cv2.putText(img, f"Detected: {class_label}", 
                   (img.shape[1] - panel_width + 20, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_INFO, 2)
        cv2.putText(img, f"Confidence: {confidence:.2f}", 
                   (img.shape[1] - panel_width + 20, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_INFO, 2)
    else:
        cv2.putText(img, "No pulling detected", 
                   (img.shape[1] - panel_width + 20, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_NORMAL, 2)
    
    # Draw tips
    cv2.putText(img, "User Tips:", 
               (img.shape[1] - panel_width + 20, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 100), 2)
    
    tips = [
        "Keep your hands visible to the camera",
        "Try to maintain good posture",
        "Take deep breaths when alerted",
        "Remember your progress goals"
    ]
    
    y_offset = 340
    for tip in tips:
        cv2.putText(img, f"- {tip}", 
                   (img.shape[1] - panel_width + 30, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        y_offset += 30

def main():
    """Main function to run the webcam feed and model inference."""
    print("Starting raw model display. Press 'q' to exit.")

    # Performance variables
    prev_time = time.time()
    fps = 0

    global last_speech_time
    global pulling_start_time

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Unable to read frame")
            break

        # Create a copy of the frame to draw on
        annotated_frame = frame.copy()

        # Execute the model on the frame
        results = model(frame, verbose=False)

        # --- Display only the highest confidence bounding box ---
        highest_conf_box = None
        max_conf = -1
        class_label = "None"
        confidence = 0.0

        # Check if any detections are present
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Iterate through all detected boxes to find the one with highest confidence
            for box in results[0].boxes:
                if box.conf is not None and box.conf.item() > max_conf:
                    max_conf = box.conf.item()
                    highest_conf_box = box
            
            # If a box with highest confidence is found, draw it
            if highest_conf_box is not None:
                x1, y1, x2, y2 = map(int, highest_conf_box.xyxy[0])
                confidence = highest_conf_box.conf.item()
                class_id = highest_conf_box.cls.item()
                class_label = model.names[int(class_id)] if model.names is not None else f"Class {int(class_id)}"
                
                # Determine box color: Red for 'pulling', Green otherwise
                box_color = COLOR_NORMAL  # Green for not pulling
                if int(class_id) == 0: 
                    box_color = COLOR_ALERT  # Red for pulling
                
                # Draw rectangle
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)
                
                # Draw label background
                label_bg_size = (200, 40)
                cv2.rectangle(annotated_frame, 
                             (x1, y1 - label_bg_size[1]), 
                             (x1 + label_bg_size[0], y1), 
                             box_color, -1)
                
                # Put label and confidence
                text = f"{class_label}: {confidence:.2f}"
                cv2.putText(annotated_frame, text, (x1 + 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # --- End of display highest confidence bounding box ---

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # --- UI Enhancements ---
        # Determine status
        status = "normal"
        
        # --- "Pulling" Detection and Text-to-Speech Trigger ---
        pulling_detected_this_frame = False
        # Check if the highest confidence detection is the "pulling" class
        # AND if its confidence is above the MIN_PULLING_CONFIDENCE threshold
        if highest_conf_box is not None and \
           int(highest_conf_box.cls.item()) == 0 and \
           highest_conf_box.conf.item() >= MIN_PULLING_CONFIDENCE: 
            pulling_detected_this_frame = True
            status = "pulling"

        # Logic for continuous pulling detection and speech trigger
        if pulling_detected_this_frame:
            if pulling_start_time is None:
                pulling_start_time = current_time
        else:
            pulling_start_time = None

        # Calculate pulling progress
        pulling_progress = 0.0
        if pulling_start_time is not None:
            pulling_progress = min(1.0, (current_time - pulling_start_time) / PULLING_DETECTION_THRESHOLD_DURATION)
            if pulling_progress > 0.5:
                status = "warning"
        
        # Check if pulling has been continuous for the threshold duration
        # AND if the speech cooldown has passed
        if pulling_start_time is not None and \
           (current_time - pulling_start_time >= PULLING_DETECTION_THRESHOLD_DURATION) and \
           (current_time - last_speech_time >= COOLDOWN_DURATION):
            
            message = random.choice(encouraging_messages)
            print(f"Sustained pulling detected! Triggering speech: '{message}'")
            speech_thread = threading.Thread(target=speak_message, args=(message,))
            speech_thread.start()
            
            last_speech_time = current_time
            pulling_start_time = None # Reset after speech to require a new sustained detection
        
        # Calculate cooldown remaining
        cooldown_remaining = max(0.0, COOLDOWN_DURATION - (current_time - last_speech_time))
        
        # Draw UI elements
        draw_ui_header(annotated_frame, status)
        draw_ui_footer(annotated_frame, cooldown_remaining)
        draw_info_panel(annotated_frame, fps, confidence, class_label)
        
        # Draw progress bars
        draw_progress_bar(annotated_frame, pulling_progress, (20, annotated_frame.shape[0] - 60), 
                         (400, 20), COLOR_PROGRESS)
        cv2.putText(annotated_frame, "Pulling Progress", 
                   (20, annotated_frame.shape[0] - 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cooldown_progress = 1.0 - (cooldown_remaining / COOLDOWN_DURATION)
        draw_progress_bar(annotated_frame, cooldown_progress, (annotated_frame.shape[1] - 420, annotated_frame.shape[0] - 60), 
                         (400, 20), COLOR_PROGRESS)
        cv2.putText(annotated_frame, "Cooldown Progress", 
                   (annotated_frame.shape[1] - 420, annotated_frame.shape[0] - 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw status indicators
        draw_status_box(annotated_frame, "Detection Active", 
                       COLOR_NORMAL if confidence > 0 else (100, 100, 100),
                       (20, 80), (200, 50))
        
        draw_status_box(annotated_frame, "System Ready", 
                       COLOR_NORMAL if cooldown_remaining == 0 else (100, 100, 100),
                       (240, 80), (200, 50))

        cv2.imshow("Trichotillomania Monitoring System", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Raw model display ended.")

if __name__ == "__main__":
    main()