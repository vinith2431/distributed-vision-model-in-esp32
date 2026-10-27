// Node 2 Pin Scanner - Round 2
// Drives each pin HIGH for 1.5s so Node 3 scanner can detect which GPIO it arrives on.
// Covers ALL 4 SPI pins on the Node2<->Node3 link: SCLK(14), MOSI(13), MISO(25), CS(26)
// Also tests GPIO27 in case MISO is actually there.
#include <Arduino.h>

// Pins to drive HIGH one at a time
// We test the 4 expected pins + 27 (already known MISO candidate)
static const int DRIVE_PINS[] = {14, 13, 25, 26, 27};
static const int N = sizeof(DRIVE_PINS) / sizeof(DRIVE_PINS[0]);

void setup() {
  Serial.begin(115200);
  delay(500);
  for (int i = 0; i < N; i++) {
    pinMode(DRIVE_PINS[i], OUTPUT);
    digitalWrite(DRIVE_PINS[i], LOW);
  }
  Serial.println("NODE2_SCANNER2_READY");
}

void loop() {
  for (int i = 0; i < N; i++) {
    int pin = DRIVE_PINS[i];
    Serial.printf("NODE2_DRIVING_HIGH: GPIO%d\n", pin);
    digitalWrite(pin, HIGH);
    delay(1500);
    digitalWrite(pin, LOW);
    delay(300);
  }
  Serial.println("NODE2_SCANNER2_CYCLE_DONE");
  delay(1000);
}
