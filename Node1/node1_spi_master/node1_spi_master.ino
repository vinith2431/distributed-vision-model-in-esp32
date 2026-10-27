#include <Arduino.h>
#include <SPI.h>

// Node 1 is SPI master toward Node 2 and USB-serial endpoint for the laptop.
static constexpr int PIN_SCLK = 18;
static constexpr int PIN_MISO = 19;
static constexpr int PIN_MOSI = 23;
static constexpr int PIN_CS   = 5;
static constexpr size_t FRAME_SIZE = 128;
static constexpr uint32_t SPI_HZ = 1000000;

SPIClass chainSPI(VSPI);
uint8_t txFrame[FRAME_SIZE];
uint8_t rxFrame[FRAME_SIZE];

uint32_t crc32_ieee(const uint8_t *data, size_t len) {
  uint32_t crc = 0xFFFFFFFFUL;
  for (size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (uint8_t b = 0; b < 8; ++b) crc = (crc & 1) ? ((crc >> 1) ^ 0xEDB88320UL) : (crc >> 1);
  }
  return crc ^ 0xFFFFFFFFUL;
}

String crcHex(const String &s) {
  char out[9];
  snprintf(out, sizeof(out), "%08lX", (unsigned long)crc32_ieee((const uint8_t*)s.c_str(), s.length()));
  return String(out);
}

void makeFrame(const String &s, uint8_t *dst) {
  memset(dst, 0, FRAME_SIZE);
  size_t n = s.length();
  if (n >= FRAME_SIZE) n = FRAME_SIZE - 1;
  memcpy(dst, s.c_str(), n);
}

String frameText(const uint8_t *src) {
  size_t n = 0;
  while (n < FRAME_SIZE && src[n] != 0) ++n;
  String s;
  s.reserve(n);
  for (size_t i = 0; i < n; ++i) s += (char)src[i];
  s.trim();
  return s;
}

String spiExchange(const String &outgoing) {
  makeFrame(outgoing, txFrame);
  memset(rxFrame, 0, sizeof(rxFrame));
  chainSPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE1));
  digitalWrite(PIN_CS, LOW);
  delayMicroseconds(10);
  chainSPI.transferBytes(txFrame, rxFrame, FRAME_SIZE);
  delayMicroseconds(10);
  digitalWrite(PIN_CS, HIGH);
  chainSPI.endTransaction();
  return frameText(rxFrame);
}

bool isFinalReply(const String &s) { return s.startsWith("ACK|") || s.startsWith("NACK|"); }

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1000);
  pinMode(PIN_CS, OUTPUT);
  digitalWrite(PIN_CS, HIGH);
  chainSPI.begin(PIN_SCLK, PIN_MISO, PIN_MOSI, PIN_CS);
  delay(300);
  Serial.println("NODE1_READY");
}

void loop() {
  if (!Serial.available()) { delay(5); return; }
  String packet = Serial.readStringUntil('\n');
  packet.trim();
  if (!packet.startsWith("PKT|")) {
    Serial.println("ERROR: expected PKT|sequence|payload|CRC32");
    return;
  }

  Serial.print("Sending to Node 2: ");
  Serial.println(packet);
  String firstReply = spiExchange(packet);
  Serial.print("Node 2 first reply: ");
  Serial.println(firstReply.length() ? firstReply : "<empty>");

  const uint32_t start = millis();
  while (millis() - start < 15000) {
    String reply = spiExchange("POLL");
    if (isFinalReply(reply)) {
      Serial.print("Response: ");
      Serial.println(reply);
      return;
    }
    delay(20);
  }
  Serial.println("TIMEOUT: no ACK/NACK from Node 2. Check Node 2/3 firmware and SPI links.");
}
