/***************************
  Simple Example Sketch for the SparkFun MEMS Microphone Breakout Board

**************************/

// Connect the MEMS AUD output to the Arduino A0 pin
int mic = 6;

// Variable to hold analog values from mic
int micOut;

void setup() {
  Serial.begin(4800);

}

void loop() {
  // read the input on analog pin 0:
  micOut = analogRead(mic);

  // print out the value you read:
  Serial.println(micOut);

}
