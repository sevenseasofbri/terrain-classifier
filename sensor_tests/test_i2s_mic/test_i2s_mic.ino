#include <driver/i2s.h>

void setup() {
  Serial.begin(115200);

  // Configure the I2S peripheral
  i2s_config_t i2s_config = {
//    .mode = I2S_MODE_MASTER | I2S_MODE_RX,
    .mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S_MSB,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  // Pin configuration
  i2s_pin_config_t pin_config = {
    .bck_io_num = 6,   // BCLK 26,
    .ws_io_num = 8,    // LRCL 22,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = 7  // DOUT 25 
  };

  // Install and start I2S driver
  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
}

void loop() {
  int32_t sample;
  size_t bytes_read;
  i2s_read(I2S_NUM_0, &sample, sizeof(sample), &bytes_read, portMAX_DELAY);

  if (bytes_read > 0) {
    Serial.println(sample);
  }
}
