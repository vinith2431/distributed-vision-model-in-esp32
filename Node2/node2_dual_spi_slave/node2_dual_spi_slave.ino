#include <Arduino.h>
#include "driver/spi_slave.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

// Node 2 is an SPI slave on BOTH independent buses.
// Bus A: Node 1 master <-> Node 2 slave (SPI3_HOST / VSPI).
// Bus B: Node 3 master <-> Node 2 slave (SPI2_HOST / HSPI).
static constexpr size_t FRAME_SIZE = 128;

// Node 1 <-> Node 2
static constexpr int N1_SCLK = 18, N1_MOSI = 23, N1_MISO = 19, N1_CS = 5;
// Node 2 <-> Node 3
static constexpr int N3_SCLK = 14, N3_MOSI = 13, N3_MISO = 25, N3_CS = 26;

static SemaphoreHandle_t stateMutex;
static char pendingPacket[FRAME_SIZE] = {0};
static char finalReply[FRAME_SIZE] = {0};

static String readFrame(const uint8_t *buf) {
  size_t n = 0;
  while (n < FRAME_SIZE && buf[n] != 0) ++n;
  String s;
  s.reserve(n);
  for (size_t i = 0; i < n; ++i) s += (char)buf[i];
  s.trim();
  return s;
}

static void writeFrame(const String &s, uint8_t *buf) {
  memset(buf, 0, FRAME_SIZE);
  size_t n = s.length();
  if (n >= FRAME_SIZE) n = FRAME_SIZE - 1;
  memcpy(buf, s.c_str(), n);
}

static void snapshotState(String &packet, String &reply) {
  xSemaphoreTake(stateMutex, portMAX_DELAY);
  packet = String(pendingPacket);
  reply = String(finalReply);
  xSemaphoreGive(stateMutex);
}

static void node1BusTask(void *) {
  alignas(4) uint8_t rx[FRAME_SIZE];
  alignas(4) uint8_t tx[FRAME_SIZE];
  while (true) {
    String packet, reply;
    snapshotState(packet, reply);
    writeFrame(reply.length() ? reply : "WAIT", tx);
    memset(rx, 0, sizeof(rx));
    spi_slave_transaction_t t = {};
    t.length = FRAME_SIZE * 8;
    t.tx_buffer = tx;
    t.rx_buffer = rx;
    esp_err_t err = spi_slave_transmit(SPI3_HOST, &t, portMAX_DELAY);
    if (err != ESP_OK) {
      Serial.printf("NODE2_BUS_N1_ERR=%d\n", (int)err);
      continue;
    }
    String incoming = readFrame(rx);
    if (incoming.startsWith("PKT|")) {
      xSemaphoreTake(stateMutex, portMAX_DELAY);
      strncpy(pendingPacket, incoming.c_str(), FRAME_SIZE - 1);
      pendingPacket[FRAME_SIZE - 1] = 0;
      finalReply[0] = 0;
      xSemaphoreGive(stateMutex);
      Serial.print("NODE2_RX_FROM_NODE1: ");
      Serial.println(incoming);
    } else if (incoming == "POLL") {
      // Normal poll from Node 1 waiting for final ACK
    } else if (incoming.length() && incoming != "WAIT") {
      Serial.print("NODE2_N1_UNEXPECTED: ");
      Serial.println(incoming);
    }
  }
}

static void node3BusTask(void *) {
  alignas(4) uint8_t rx[FRAME_SIZE];
  alignas(4) uint8_t tx[FRAME_SIZE];
  uint32_t txCount = 0;
  while (true) {
    String packet, reply;
    snapshotState(packet, reply);
    String toSend = packet.length() ? packet : "WAIT";
    writeFrame(toSend, tx);
    memset(rx, 0, sizeof(rx));
    spi_slave_transaction_t t = {};
    t.length = FRAME_SIZE * 8;
    t.tx_buffer = tx;
    t.rx_buffer = rx;

    esp_err_t err = spi_slave_transmit(SPI2_HOST, &t, portMAX_DELAY);
    if (err != ESP_OK) {
      Serial.printf("NODE2_BUS_N3_ERR=%d\n", (int)err);
      continue;
    }

    txCount++;
    String incoming = readFrame(rx);
    // Log every 20th transaction to show the bus is alive, or log notable ones
    if (txCount % 20 == 1 || incoming.startsWith("ACK|") || incoming.startsWith("NACK|") || incoming.startsWith("PKT|")) {
      Serial.printf("NODE2_N3_TX#%lu: sent='%s' rx='%s'\n", (unsigned long)txCount,
                    toSend.substring(0, 20).c_str(), incoming.substring(0, 20).c_str());
    }

    if (incoming.startsWith("ACK|") || incoming.startsWith("NACK|")) {
      xSemaphoreTake(stateMutex, portMAX_DELAY);
      strncpy(finalReply, incoming.c_str(), FRAME_SIZE - 1);
      finalReply[FRAME_SIZE - 1] = 0;
      pendingPacket[0] = 0;
      xSemaphoreGive(stateMutex);
      Serial.print("NODE2_RX_FROM_NODE3: ");
      Serial.println(incoming);
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);

  stateMutex = xSemaphoreCreateMutex();
  if (!stateMutex) {
    Serial.println("NODE2_FATAL: mutex create failed");
    while (true) delay(1000);
  }

  // SPI3_HOST (VSPI) for Node 1
  spi_bus_config_t bus1 = {};
  bus1.mosi_io_num = N1_MOSI;
  bus1.miso_io_num = N1_MISO;
  bus1.sclk_io_num = N1_SCLK;
  bus1.quadwp_io_num = -1;
  bus1.quadhd_io_num = -1;
  bus1.max_transfer_sz = FRAME_SIZE;

  spi_slave_interface_config_t dev1 = {};
  dev1.mode = 1;
  dev1.spics_io_num = N1_CS;
  dev1.queue_size = 1;
  esp_err_t e1 = spi_slave_initialize(SPI3_HOST, &bus1, &dev1, SPI_DMA_CH_AUTO);

  // SPI2_HOST (HSPI) for Node 3
  spi_bus_config_t bus3 = {};
  bus3.mosi_io_num = N3_MOSI;
  bus3.miso_io_num = N3_MISO;
  bus3.sclk_io_num = N3_SCLK;
  bus3.quadwp_io_num = -1;
  bus3.quadhd_io_num = -1;
  bus3.max_transfer_sz = FRAME_SIZE;

  spi_slave_interface_config_t dev3 = {};
  dev3.mode = 1;
  dev3.spics_io_num = N3_CS;
  dev3.queue_size = 1;
  esp_err_t e3 = spi_slave_initialize(SPI2_HOST, &bus3, &dev3, SPI_DMA_CH_AUTO);

  if (e1 != ESP_OK || e3 != ESP_OK) {
    Serial.printf("NODE2_FATAL: SPI init errors bus1=%d bus3=%d\n", (int)e1, (int)e3);
    while (true) delay(1000);
  }
  Serial.println("NODE2_READY");
  xTaskCreatePinnedToCore(node1BusTask, "node1_spi_slave", 4096, nullptr, 2, nullptr, 0);
  xTaskCreatePinnedToCore(node3BusTask, "node3_spi_slave", 4096, nullptr, 2, nullptr, 1);
}

void loop() {
  delay(1000);
}
