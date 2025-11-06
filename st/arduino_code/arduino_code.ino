/*
  AnalogReadSerial

  Reads an analog input on pin 0, prints the result to the Serial Monitor.
  Attach the output pin of a EMG sensor to pin A0, and the outside pins to +5V and ground.

  This example code is in the public domain.

  https://www.arduino.cc/en/Tutorial/BuiltInExamples/AnalogReadSerial
*/
int data;

void setup() {
  Serial.begin(115200);
  pinMode(7, OUTPUT);
  pinMode(5,OUTPUT);
}

void loop() {

  {
    data = Serial.read();
    if (data == '1')
    digitalWrite(7, HIGH);    // turn on LED1 
    else if (data == '2')
    digitalWrite(5, HIGH);    //turn on LED2
    else if (data == '0') 
    { digitalWrite(7, LOW);
      digitalWrite(5,LOW);}
  }

  float sensorValue = analogRead(A0);

  Serial.println(sensorValue);

  delay(100); // Wait 100ms
}
