const int ledPin=13;
void setup() {
    Serial.begin(9600);
    pinMode(ledPin,OUTPUT);
    pinMode(6,INPUT);
}

void loop() {
    int sensorState = analogRead(6);
    Serial.println(sensorState);
    delay(100);
    if(sensorState == HIGH)
    {
        digitalWrite(ledPin,HIGH);
//        Serial.println('Vibration Detected!');
    }
    else
    {
        digitalWrite(ledPin,LOW);
//        Serial.println('No Vibration');
    }
}
