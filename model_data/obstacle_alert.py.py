#!/usr/bin/env python3
import time
import RPi.GPIO as GPIO
import pygame

# ------------ CONFIGURATION ------------ #

# GPIO pins for sensors
LEFT_TRIG_PIN  = 23   # GPIO 23 (Pin 16)
LEFT_ECHO_PIN  = 24   # GPIO 24 (Pin 18)
RIGHT_TRIG_PIN = 17   # GPIO 17 (Pin 11)
RIGHT_ECHO_PIN = 27   # GPIO 27 (Pin 13)

# Distance threshold (in cm) at which to trigger warning
THRESHOLD_CM = 50

# Paths to directional audio files (stereo WAVs)
LEFT_SOUND_PATH  = '/home/ubuntu/obstacle_alert/sounds/caution_left.wav'
RIGHT_SOUND_PATH = '/home/ubuntu/obstacle_alert/sounds/caution_right.wav'

# Minimum interval between successive warnings (seconds)
COOLDOWN_SECONDS = 1.2

# --------------------------------------- #

def init_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Set up left sensor pins
    GPIO.setup(LEFT_TRIG_PIN, GPIO.OUT)
    GPIO.setup(LEFT_ECHO_PIN, GPIO.IN)
    # Set trigger low initially
    GPIO.output(LEFT_TRIG_PIN, GPIO.LOW)

    # Set up right sensor pins
    GPIO.setup(RIGHT_TRIG_PIN, GPIO.OUT)
    GPIO.setup(RIGHT_ECHO_PIN, GPIO.IN)
    # Set trigger low initially
    GPIO.output(RIGHT_TRIG_PIN, GPIO.LOW)

    # Allow sensors to settle
    time.sleep(2)

def measure_distance(trig_pin, echo_pin):
    """
    Sends a 10µs pulse on trig_pin, listens on echo_pin, and returns distance (cm).
    Returns None if timeout.
    """
    # Ensure trigger is low
    GPIO.output(trig_pin, GPIO.LOW)
    time.sleep(0.0002)  # 200µs

    # Send 10µs HIGH pulse
    GPIO.output(trig_pin, GPIO.HIGH)
    time.sleep(0.00001)  # 10µs
    GPIO.output(trig_pin, GPIO.LOW)

    # Wait for echo to go HIGH
    timeout_start = time.time()
    while GPIO.input(echo_pin) == 0:
        if time.time() - timeout_start > 0.02:  # 20ms timeout
            return None
    start_time = time.time()

    # Wait for echo to go LOW
    timeout_start = time.time()
    while GPIO.input(echo_pin) == 1:
        if time.time() - timeout_start > 0.02:  # 20ms timeout
            return None
    end_time = time.time()

    # Time difference in seconds
    delta = end_time - start_time
    # Convert to distance (speed of sound = 34300 cm/s, divide by 2 for round-trip)
    distance_cm = (delta * 34300) / 2
    return distance_cm

def init_audio():
    pygame.mixer.init()
    # Preload sounds
    left_sound  = pygame.mixer.Sound(LEFT_SOUND_PATH)
    right_sound = pygame.mixer.Sound(RIGHT_SOUND_PATH)
    return left_sound, right_sound

def main():
    try:
        init_gpio()
        left_sound, right_sound = init_audio()
        last_warning_time = 0

        print("Obstacle detection running. Threshold: {} cm".format(THRESHOLD_CM))
        while True:
            # Measure left and right distances
            dist_left  = measure_distance(LEFT_TRIG_PIN, LEFT_ECHO_PIN)
            dist_right = measure_distance(RIGHT_TRIG_PIN, RIGHT_ECHO_PIN)

            now = time.time()
            # If enough time has passed since last warning
            if now - last_warning_time >= COOLDOWN_SECONDS:
                # Check LEFT sensor first
                if dist_left is not None and dist_left <= THRESHOLD_CM:
                    print(f"[{time.strftime('%H:%M:%S')}] Obstacle detected on LEFT: {dist_left:.1f} cm")
                    left_sound.play()
                    last_warning_time = now
                # Else check RIGHT sensor
                elif dist_right is not None and dist_right <= THRESHOLD_CM:
                    print(f"[{time.strftime('%H:%M:%S')}] Obstacle detected on RIGHT: {dist_right:.1f} cm")
                    right_sound.play()
                    last_warning_time = now

            # Optional: print distances for debugging every second
            print(f"  Left:  {dist_left if dist_left else '-'} cm | Right: {dist_right if dist_right else '–'} cm", end="\r")

            time.sleep(0.1)  # 100 ms loop delay

    except KeyboardInterrupt:
        print("\nExiting due to user interrupt.")
    finally:
        GPIO.cleanup()
        pygame.mixer.quit()

if __name__ == "__main__":
    main()
