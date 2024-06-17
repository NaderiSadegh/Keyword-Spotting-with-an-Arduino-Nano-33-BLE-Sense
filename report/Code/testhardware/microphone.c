////////////////////////////////////////
// Author: Malik Al Ashter Ghansletwala
// Date added: 8.02.2024
// Path: ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense\report\Code\testhardware
// Version: 2
// Reviewed by: Malik Al Ashter Ghansletwala
// Review Date: 9.02.2024
////////////////////////////////////////

#include <Arduino_LSM9DS1.h>
#include <PDM.h>
#include <Keywordspotting.h> 

// RGB LED pins on the Arduino Nano 33 BLE Sense
const int RED_LED = 22;
const int GREEN_LED = 23;
const int BLUE_LED = 24;

// Function to blink the LED with the specified color
void blinkLED(int red, int green, int blue) {
  digitalWrite(RED_LED, red);
  digitalWrite(GREEN_LED, green);
  digitalWrite(BLUE_LED, blue);
  delay(500);
  digitalWrite(RED_LED, LOW);
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(BLUE_LED, LOW);
  delay(500);
}

void setup() {
  // Initialize serial communication
  Serial.begin(115200);

  // Initialize the RGB LED pins
  pinMode(RED_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(BLUE_LED, OUTPUT);

  // Initialize Edge Impulse model
  if (!Keywordspotting.begin()) {
    Serial.println("Failed to initialize Edge Impulse model");
    while (1);
  }
}

void loop() {
  // Run the machine learning model
  ei_impulse_result_t result = { 0 };
  if (Keywordspotting.run_classifier(&result, false) != EI_IMPULSE_OK) {
    Serial.println("Failed to run classifier");
    return;
  }

  // Check the classification results and blink the LED accordingly
  if (result.classification[0].value > 0.8) { // Assuming "yes" is at index 0
    blinkLED(LOW, HIGH, LOW); // Blink green for "yes"
  } else if (result.classification[1].value > 0.8) { // Assuming "no" is at index 1
    blinkLED(HIGH, LOW, LOW); // Blink red for "no"
  } else {
    blinkLED(LOW, LOW, HIGH); // Blink blue for unrecognized words
  }
}
