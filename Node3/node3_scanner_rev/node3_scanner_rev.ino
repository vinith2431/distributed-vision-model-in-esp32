// Node 3 drives its SPI master output pins HIGH one at a time.
// SCLK(14), MOSI(13), CS(26) — Node 3 is master, these are outputs toward Node 2.
// Also drives 27 (physical MISO landing) as a cross-check.
#include <Arduino.h>

static const int DRIVE_PINS[] = {14, 13, 26, 27};
static const int N = sizeof(DRIVE_PINS) / sizeof(DRIVE_PINS[0]);

void setup() {
  Serial.begin(115200);
  delay(500);
  for (int i = 0; i < N; i++) {
    pinMode(DRIVE_PINS[i], OUTPUT);
    digitalWrite(DRIVE_PINS[i], LOW);
  }
  Serial.println("NODE3_SCANNER_READY");
}

void loop() {
  for (int i = 0; i < N; i++) {
    int pin = DRIVE_PINS[i];
    Serial.printf("NODE3_DRIVING_HIGH: GPIO%d\n", pin);
    digitalWrite(pin, HIGH);
    delay(1500);
    digitalWrite(pin, LOW);
    delay(300);
  }
  Serial.println("NODE3_SCANNER_CYCLE_DONE");
  delay(1000);
}
