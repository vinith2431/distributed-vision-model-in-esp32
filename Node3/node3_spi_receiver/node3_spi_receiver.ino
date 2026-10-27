#include <Arduino.h>
#include <SPI.h>

static const int PIN_SCLK = 14;
static const int PIN_MISO = 25;
static const int PIN_MOSI = 13;
static const int PIN_CS   = 26;

static const size_t FRAME_SIZE = 128;

SPIClass link(HSPI);
uint8_t txFrame[FRAME_SIZE];
uint8_t rxFrame[FRAME_SIZE];

String pendingResponse;

uint32_t crc32(const uint8_t *data, size_t len) {
  uint32_t crc = 0xFFFFFFFF;

  for (size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (int b = 0; b < 8; ++b) {
      crc = (crc & 1)
          ? (crc >> 1) ^ 0xEDB88320
          : (crc >> 1);
    }
  }

  return crc ^ 0xFFFFFFFF;
}

String crcHex(const String &s) {
  char out[9];
  snprintf(out, sizeof(out), "%08X",
           (unsigned int)crc32(
             (const uint8_t *)s.c_str(), s.length()));
  return String(out);
}

String receivedText() {
  size_t n = 0;
  while (n < FRAME_SIZE && rxFrame[n] != 0) ++n;

  String s;
  s.reserve(n);
  for (size_t i = 0; i < n; ++i) s += (char)rxFrame[i];
  s.trim();
  return s;
}

String exchange(const String &message) {
  memset(txFrame, 0, sizeof(txFrame));
  memset(rxFrame, 0, sizeof(rxFrame));

  size_t n = min(message.length(), FRAME_SIZE - 1);
  memcpy(txFrame, message.c_str(), n);

  link.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));
  digitalWrite(PIN_CS, LOW);
  link.transferBytes(txFrame, rxFrame, FRAME_SIZE);
  digitalWrite(PIN_CS, HIGH);
  link.endTransaction();

  return receivedText();
}

String makeResponse(String kind, String sequence, String status) {
  String body = sequence + "|" + status;
  return kind + "|" + body + "|" + crcHex(body);
}

void processPacket(const String &packet) {
  int a = packet.indexOf('|');
  int b = packet.indexOf('|', a + 1);
  int c = packet.lastIndexOf('|');

  if (a < 0 || b < 0 || c <= b ||
      packet.substring(0, a) != "PKT") {
    pendingResponse = makeResponse("NACK", "0", "FORMAT");
    Serial.println("NODE3: FORMAT_ERROR");
    return;
  }

  String sequence = packet.substring(a + 1, b);
  String payload = packet.substring(b + 1, c);
  String gotCRC = packet.substring(c + 1);

  String body = sequence + "|" + payload;

  if (!gotCRC.equalsIgnoreCase(crcHex(body))) {
    pendingResponse = makeResponse("NACK", sequence, "BADCRC");
    Serial.println("NODE3: CRC_FAIL");
    return;
  }

  Serial.print("NODE3: CRC_OK, payload=");
  Serial.println(payload);

  pendingResponse = makeResponse("ACK", sequence, "OK");
  Serial.println("NODE3: ACK_PREPARED");
}

void setup() {
  Serial.begin(115200);

  pinMode(PIN_CS, OUTPUT);
  digitalWrite(PIN_CS, HIGH);

  link.begin(PIN_SCLK, PIN_MISO, PIN_MOSI, PIN_CS);

  Serial.println("NODE3_READY");
}

void loop() {
  // Node 3 is master on this link and polls Node 2.
  String command = pendingResponse.length()
      ? pendingResponse
      : "POLL";

  String response = exchange(command);

  if (pendingResponse.length() && command == pendingResponse) {
    // ACK/NACK has just been sent to Node 2.
    pendingResponse = "";
    Serial.println("NODE3: RESPONSE_SENT_TO_NODE2");
  }

  if (command == "POLL" && response.startsWith("PKT|")) {
    processPacket(response);
  }

  delay(10);
}
