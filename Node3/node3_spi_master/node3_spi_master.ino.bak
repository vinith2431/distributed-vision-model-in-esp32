#include <Arduino.h>
#include <SPI.h>

// Node 3 is SPI master toward Node 2. It polls for a packet, verifies CRC32,
// and returns ACK/NACK in the following SPI transaction.
static constexpr int PIN_SCLK = 14;
static constexpr int PIN_MISO = 25;
static constexpr int PIN_MOSI = 13;
static constexpr int PIN_CS = 26;
static constexpr size_t FRAME_SIZE = 128;
static constexpr uint32_t SPI_HZ = 1000000;
SPIClass chainSPI(HSPI);
uint8_t txFrame[FRAME_SIZE], rxFrame[FRAME_SIZE];
String responseToSend = "POLL";
String lastProcessedPacket = "";

uint32_t crc32_ieee(const uint8_t *data, size_t len) {
  uint32_t crc = 0xFFFFFFFFUL;
  for (size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (uint8_t b = 0; b < 8; ++b) crc = (crc & 1) ? ((crc >> 1) ^ 0xEDB88320UL) : (crc >> 1);
  }
  return crc ^ 0xFFFFFFFFUL;
}
String crcHex(const String &s) {
  char out[9]; snprintf(out, sizeof(out), "%08lX", (unsigned long)crc32_ieee((const uint8_t*)s.c_str(), s.length()));
  return String(out);
}
void makeFrame(const String &s, uint8_t *dst) {
  memset(dst, 0, FRAME_SIZE); size_t n = s.length(); if (n >= FRAME_SIZE) n = FRAME_SIZE - 1; memcpy(dst, s.c_str(), n);
}
String frameText(const uint8_t *src) {
  size_t n = 0; while (n < FRAME_SIZE && src[n] != 0) ++n;
  String s; s.reserve(n); for (size_t i = 0; i < n; ++i) s += (char)src[i]; s.trim(); return s;
}
String makeNack(const String &seq, const String &status) {
  String body = seq + "|" + status;
  return "NACK|" + body + "|" + crcHex(body);
}
String validatePacket(const String &p) {
  if (!p.startsWith("PKT|")) return "";
  int a = p.indexOf('|'); int b = p.indexOf('|', a + 1); int c = p.indexOf('|', b + 1);
  if (a < 0 || b < 0 || c < 0 || p.indexOf('|', c + 1) >= 0) return makeNack("0", "FORMAT");
  String seq = p.substring(a + 1, b), payload = p.substring(b + 1, c), supplied = p.substring(c + 1);
  if (!seq.length() || !payload.length() || supplied.length() != 8) return makeNack(seq.length() ? seq : "0", "FORMAT");
  String body = seq + "|" + payload;
  if (!supplied.equalsIgnoreCase(crcHex(body))) return makeNack(seq, "BADCRC");
  String ackBody = seq + "|OK";
  return "ACK|" + ackBody + "|" + crcHex(ackBody);
}
String spiExchange(const String &outgoing) {
  makeFrame(outgoing, txFrame); memset(rxFrame, 0, sizeof(rxFrame));
  chainSPI.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE0));
  digitalWrite(PIN_CS, LOW); chainSPI.transferBytes(txFrame, rxFrame, FRAME_SIZE); digitalWrite(PIN_CS, HIGH);
  chainSPI.endTransaction(); return frameText(rxFrame);
}
void setup() {
  Serial.begin(115200); pinMode(PIN_CS, OUTPUT); digitalWrite(PIN_CS, HIGH);
  chainSPI.begin(PIN_SCLK, PIN_MISO, PIN_MOSI, PIN_CS); delay(300);
  Serial.println("NODE3_READY");
}
void loop() {
  String received = spiExchange(responseToSend);
  responseToSend = "POLL";
  if (received.startsWith("PKT|") && received != lastProcessedPacket) {
    Serial.print("NODE3_PACKET: "); Serial.println(received);
    responseToSend = validatePacket(received);
    if (!responseToSend.length()) responseToSend = makeNack("0", "FORMAT");
    lastProcessedPacket = received;
    Serial.print("NODE3_RESULT: "); Serial.println(responseToSend);
  } else if (received == "WAIT" || received == "POLL" || received.length() == 0) {
    // No packet currently available.
  } else if (received.startsWith("PKT|") && received == lastProcessedPacket) {
    // Node 2 may keep offering the packet until it receives the queued reply.
    responseToSend = validatePacket(received);
  } else if (received.startsWith("ACK|") || received.startsWith("NACK|")) {
    // Node 2 should not send a final response toward Node 3.
  }
  delay(10);
}
