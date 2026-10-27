// Node 3 Pin Listener - Round 2
// Listens on a wide set of GPIOs with INPUT_PULLDOWN and reports which one goes HIGH.
// Used to map ALL 4 SPI wires from Node 2.
#include <Arduino.h>

// All candidate GPIOs to listen on - covers common ESP32 GPIO range + the SPI candidates
static const int LISTEN_PINS[] = {4, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33, 34, 35};
static const int N = sizeof(LISTEN_PINS) / sizeof(LISTEN_PINS[0]);

void setup() {
  Serial.begin(115200);
  delay(500);
  for (int i = 0; i < N; i++) {
    int pin = LISTEN_PINS[i];
    if (pin < 34) {  // 34, 35 are input-only, no pull config needed on some variants
      pinMode(pin, INPUT_PULLDOWN);
    } else {
      pinMode(pin, INPUT);
    }
  }
  Serial.println("NODE3_LISTENER2_READY");
}

void loop() {
  for (int i = 0; i < N; i++) {
    int pin = LISTEN_PINS[i];
    if (digitalRead(pin) == HIGH) {
      Serial.printf("NODE3_DETECTED_HIGH: GPIO%d\n", pin);
      delay(200);  // Debounce - only print a few times while HIGH
    }
  }
  delay(50);
}
