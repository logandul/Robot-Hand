#include <Servo.h>

//config
const int MCP_PIN = 9;  
const int PIP_PIN = 10; 
const int DIP_PIN = 11; 
const int BAUD_RATE = 9600; 
Servo servoMCP;
Servo servoPIP;
Servo servoDIP;

void setup() {
  Serial.begin(BAUD_RATE); 
  Serial.setTimeout(10); 
  servoMCP.attach(MCP_PIN);
  servoPIP.attach(PIP_PIN);
  servoDIP.attach(DIP_PIN);
  servoMCP.write(0);
  servoPIP.write(0);
  servoDIP.write(0);

  Serial.println("Arduino Servo Controller Ready.");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); 
    command.trim(); 
    int firstComma = command.indexOf(',');
    int secondComma = command.indexOf(',', firstComma + 1);

    if (firstComma > 0 && secondComma > firstComma) {
      String mcpStr = command.substring(0, firstComma);
      String pipStr = command.substring(firstComma + 1, secondComma);
      String dipStr = command.substring(secondComma + 1);

      int angleMCP = mcpStr.toInt();
      int anglePIP = pipStr.toInt();
      int angleDIP = dipStr.toInt();

      
      angleMCP = constrain(angleMCP, 0, 180);
      anglePIP = constrain(anglePIP, 0, 180);
      angleDIP = constrain(angleDIP, 0, 180);

      
      servoMCP.write(angleMCP);
      servoPIP.write(anglePIP);
      servoDIP.write(angleDIP);
    } else {
       
    }
  }
}