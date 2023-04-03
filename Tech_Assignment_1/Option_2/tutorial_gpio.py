# Raspberry Pi GPIO Tutorial

# Import the necessary libraries
import RPi.GPIO as GPIO
import time

# Set the GPIO mode
GPIO.setmode(GPIO.BCM)

# Set the GPIO pin number
pin = 18

# Set the pin as an output
GPIO.setup(pin, GPIO.OUT)

# Set the pin to high
GPIO.output(pin, GPIO.HIGH)

# Wait for 1 second
time.sleep(1)

# Set the pin to low
GPIO.output(pin, GPIO.LOW)

# Wait for 1 second
time.sleep(1)