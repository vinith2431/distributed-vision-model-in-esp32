// Node 2 listens on its SPI2_HOST (HSPI) pins + neighbors for Node 3 signals.
// Node 3 is master: SCLK comes into Node2, MOSI comes into Node2, CS comes into Node2.
// Expected: SCLK=14, MOSI=13, CS=26 on Node 2 side.
// Also listens on surrounding pins in case wires are off.
#include <Arduino.h>

static const int LISTEN_PINS[] = {4, 12, 13, 14, 15, 21, 22, 25, 26, 27, 32, 33};
static const int N = sizeof(LISTEN_PINS) / sizeof(LISTEN_PINS[0]);

void setup() {
  Serial.begin(115200);
  delay(500);
  for (int i = 0; i < N; i++) {
    int pin = LISTEN_PINS[i];
    if (pin < 34) pinMode(pin, INPUT_PULLDOWN);
    else          pinMode(pin, INPUT);
  }
  Serial.println("NODE2_LISTENER_READY");
}

void loop() {
  for (int i = 0; i < N; i++) {
    int pin = LISTEN_PINS[i];
    if (digitalRead(pin) == HIGH) {
      Serial.printf("NODE2_DETECTED_HIGH: GPIO%d\n", pin);
      delay(200);
    }
  }
  delay(50);
}
