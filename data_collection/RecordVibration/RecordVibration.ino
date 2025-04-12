

#define EXTERNAL_ADC_PIN A11   // ADCPIN is the lowest analog capable pin exposed on the variant
                         // - if no exposed pins are analog capable this will be undefined
                                  // - to use another pin provide an analog capable pin number such as:
                                  //   - A0 -> A9 (when analog pins are named sequentially from 0)
                                  //   - A11 -> A13, A16, A29, A31 -> A35 (when pins are named after Apollo3 pads)
                                  //   - A variant-specific pin number (when none of the above apply)

void setup() {
  Serial.begin(500000);
  while (!Serial);
}


void loop() {
  uint16_t external = analogRead(EXTERNAL_ADC_PIN); // reads the analog voltage on the selected analog pin
  Serial.println(external);
  delayMicroseconds(64); // 64us == 15.625kHz sample rate
}
