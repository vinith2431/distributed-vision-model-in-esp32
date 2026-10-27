#include <Arduino.h>
#include "driver/spi_slave.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

// Node 2 is an SPI slave on BOTH independent buses.
// Bus A: Node 1 master <-> Node 2 slave. Bus B: Node 3 master <-> Node 2 slave.
static constexpr size_t FRAME_SIZE = 128;
static constexpr uint32_t SPI_HZ = 1000000;

// Node 1 <-> Node 2
static constexpr int N1_SCLK = 18, N1_MOSI = 23, N1_MISO = 19, N1_CS = 5;
// Node 2 <-> Node 3
static constexpr int N3_SCLK = 14, N3_MOSI = 13, N3_MISO = 25, N3_CS = 26;

static SemaphoreHandle_t stateMutex;
static char pendingPacket[FRAME_SIZE] = {0};
static char finalReply[FRAME_SIZE] = {0};

static String readFrame(const uint8_t *buf) {
  size_t n = 0; while (n < FRAME_SIZE && buf[n] != 0) ++n;
  String s; s.reserve(n); for (size_t i = 0; i < n; ++i) s += (char)buf[i];
  s.trim(); return s;
}
static void writeFrame(const String &s, uint8_t *buf) {
  memset(buf, 0, FRAME_SIZE); size_t n = s.length(); if (n >= FRAME_SIZE) n = FRAME_SIZE - 1;
  memcpy(buf, s.c_str(), n);
}
static void snapshotState(String &packet, String &reply) {
  xSemaphoreTake(stateMutex, portMAX_DELAY);
  packet = String(pendingPacket); reply = String(finalReply);
  xSemaphoreGive(stateMutex);
}

static esp_err_t initSlave(spi_host_device_t host, int sclk, int mosi, int miso, int cs) {
  spi_bus_config_t bus = {};
  bus.mosi_io_num = mosi; bus.miso_io_num = miso; bus.sclk_io_num = sclk;
  bus.quadwp_io_num = -1; bus.quadhd_io_num = -1; bus.max_transfer_sz = FRAME_SIZE;
  esp_err_t err = spi_slave_initialize(host, &bus, nullptr, SPI_DMA_CH_AUTO);
  if (err != ESP_OK) return err;
  spi_slave_interface_config_t dev = {};
  dev.mode = 0; dev.spics_io_num = cs; dev.queue_size = 1; dev.flags = 0;
  dev.post_setup_cb = nullptr; dev.post_trans_cb = nullptr;
  return spi_slave_initialize(host, &bus, &dev, SPI_DMA_CH_AUTO);
}

// Initialize each bus separately. Each task waits for one 128-byte SPI transaction.
static void node1BusTask(void *) {
  alignas(4) uint8_t rx[FRAME_SIZE];
  alignas(4) uint8_t tx[FRAME_SIZE];
  while (true) {
    String packet, reply; snapshotState(packet, reply);
    writeFrame(reply.length() ? reply : "WAIT", tx);
    memset(rx, 0, sizeof(rx));
    spi_slave_transaction_t t = {}; t.length = FRAME_SIZE * 8; t.tx_buffer = tx; t.rx_buffer = rx;
    esp_err_t err = spi_slave_transmit(SPI3_HOST, &t, portMAX_DELAY);
    if (err != ESP_OK) { Serial.printf("NODE2_BUS_N1_ERR=%d\n", (int)err); continue; }
    String incoming = readFrame(rx);
    if (incoming.startsWith("PKT|")) {
      xSemaphoreTake(stateMutex, portMAX_DELAY);
      strncpy(pendingPacket, incoming.c_str(), FRAME_SIZE - 1); pendingPacket[FRAME_SIZE - 1] = 0;
      finalReply[0] = 0;
      xSemaphoreGive(stateMutex);
      Serial.print("NODE2_RX_FROM_NODE1: "); Serial.println(incoming);
    } else if (incoming == "POLL") {
      // Poll is expected; no additional action.
    } else if (incoming.length() && incoming != "WAIT") {
      Serial.print("NODE2_N1_UNEXPECTED: "); Serial.println(incoming);
    }
  }
}

static void node3BusTask(void *) {
  alignas(4) uint8_t rx[FRAME_SIZE];
  alignas(4) uint8_t tx[FRAME_SIZE];
  while (true) {
    String packet, reply; snapshotState(packet, reply);
    writeFrame(packet.length() ? packet : "WAIT", tx);
    memset(rx, 0, sizeof(rx));
    spi_slave_transaction_t t = {}; t.length = FRAME_SIZE * 8; t.tx_buffer = tx; t.rx_buffer = rx;
    esp_err_t err = spi_slave_transmit(SPI2_HOST, &t, portMAX_DELAY);
    if (err != ESP_OK) { Serial.printf("NODE2_BUS_N3_ERR=%d\n", (int)err); continue; }
    String incoming = readFrame(rx);
    if (incoming.startsWith("ACK|") || incoming.startsWith("NACK|")) {
      xSemaphoreTake(stateMutex, portMAX_DELAY);
      strncpy(finalReply, incoming.c_str(), FRAME_SIZE - 1); finalReply[FRAME_SIZE - 1] = 0;
      pendingPacket[0] = 0;
      xSemaphoreGive(stateMutex);
      Serial.print("NODE2_RX_FROM_NODE3: "); Serial.println(incoming);
    } else if (incoming.length() && incoming != "POLL" && incoming != "WAIT") {
      Serial.print("NODE2_N3_UNEXPECTED: "); Serial.println(incoming);
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  stateMutex = xSemaphoreCreateMutex();
  if (!stateMutex) { Serial.println("NODE2_FATAL: mutex create failed"); while (true) delay(1000); }

  // ESP32 classic SPI2_HOST = HSPI and SPI3_HOST = VSPI. Node 1 uses VSPI; Node 3 uses HSPI.
  spi_bus_config_t bus1 = {};
  bus1.mosi_io_num = N1_MOSI; bus1.miso_io_num = N1_MISO; bus1.sclk_io_num = N1_SCLK;
  bus1.quadwp_io_num = -1; bus1.quadhd_io_num = -1; bus1.max_transfer_sz = FRAME_SIZE;
  spi_slave_interface_config_t dev1 = {};
  dev1.mode = 0; dev1.spics_io_num = N1_CS; dev1.queue_size = 1;
  esp_err_t e1 = spi_slave_initialize(SPI3_HOST, &bus1, &dev1, SPI_DMA_CH_AUTO);

  spi_bus_config_t bus3 = {};
  bus3.mosi_io_num = N3_MOSI; bus3.miso_io_num = N3_MISO; bus3.sclk_io_num = N3_SCLK;
  bus3.quadwp_io_num = -1; bus3.quadhd_io_num = -1; bus3.max_transfer_sz = FRAME_SIZE;
  spi_slave_interface_config_t dev3 = {};
  dev3.mode = 0; dev3.spics_io_num = N3_CS; dev3.queue_size = 1;
  esp_err_t e3 = spi_slave_initialize(SPI2_HOST, &bus3, &dev3, SPI_DMA_CH_AUTO);

  if (e1 != ESP_OK || e3 != ESP_OK) {
    Serial.printf("NODE2_FATAL: SPI init errors bus1=%d bus3=%d\n", (int)e1, (int)e3);
    while (true) delay(1000);
  }
  Serial.println("NODE2_READY");
  xTaskCreatePinnedToCore(node1BusTask, "node1_spi_slave", 4096, nullptr, 2, nullptr, 0);
  xTaskCreatePinnedToCore(node3BusTask, "node3_spi_slave", 4096, nullptr, 2, nullptr, 1);
}

void loop() { delay(1000); }
