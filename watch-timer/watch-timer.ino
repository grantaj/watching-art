#include <SevSegShift.h>
#include <NewPing.h>

// Display pin setup
#define SHIFT_PIN_DS   13
#define SHIFT_PIN_STCP 12
#define SHIFT_PIN_SHCP 11

// Ultrasonic sensor pins
#define TRIGGER_PIN 9
#define ECHO_PIN    10
#define MAX_DISTANCE 100
#define DETECTION_THRESHOLD 50

SevSegShift sevseg(SHIFT_PIN_DS, SHIFT_PIN_SHCP, SHIFT_PIN_STCP);
NewPing sonar(TRIGGER_PIN, ECHO_PIN, MAX_DISTANCE);

// State and timing
unsigned long detectionStart = 0;
unsigned long watchDuration = 0;
unsigned long flashStart = 0;
unsigned long potentialEnd = 0;
unsigned long lastPingTime = 0;
unsigned int currentDistance = 0;

float totalDuration = 0;
int sessionCount = 0;

bool watching = false;
bool flashing = false;

const int debounceTime = 500;
const int flashDuration = 3000;
const int pingInterval = 100;

void setup() {
  byte numDigits = 8;
  byte digitPins[] = {8, 9, 10, 11, 12, 13, 14, 15};
  byte segmentPins[] = {0, 1, 2, 3, 4, 5, 6, 7};
  bool resistorsOnSegments = false;
  byte hardwareConfig = COMMON_ANODE;
  bool updateWithDelays = false;
  bool leadingZeros = false;
  bool disableDecPoint = false;

  sevseg.begin(hardwareConfig, numDigits, digitPins, segmentPins,
               resistorsOnSegments, updateWithDelays, leadingZeros, disableDecPoint);
  sevseg.setBrightness(90);
}

void loop() {
  unsigned long now = millis();
  sevseg.refreshDisplay();

  // Blocking ping every 250ms
  if (now - lastPingTime >= pingInterval) {
    lastPingTime = now;
    currentDistance = sonar.ping_cm();
  }

  if (currentDistance > 0 && currentDistance <= DETECTION_THRESHOLD) {
    if (!watching && detectionStart == 0) {
      detectionStart = now;
    } else if (!watching && (now - detectionStart >= debounceTime)) {
      watching = true;
      potentialEnd = 0;
    }
    if (watching) {
      unsigned long liveDuration = now - detectionStart;
      sevseg.setNumber(liveDuration / 100, 1);
    }
  } else {
    if (watching && potentialEnd == 0) {
      potentialEnd = now;
    } else if (watching && (now - potentialEnd >= debounceTime)) {
      watching = false;
      flashing = true;
      flashStart = now;

      watchDuration = potentialEnd - detectionStart;
      totalDuration += watchDuration / 1000.0;
      sessionCount++;

      detectionStart = 0;
      potentialEnd = 0;
    } else if (!watching && currentDistance > DETECTION_THRESHOLD) {
      detectionStart = 0;
      potentialEnd = 0;
    }
  }

  if (flashing) {
    if ((now - flashStart) < flashDuration) {
      if ((now / 500) % 2 == 0) {
        sevseg.blank();
      } else {
        sevseg.setNumber(watchDuration / 100, 1);
      }
    } else {
      flashing = false;
    }
  } else if (!watching && !flashing) {
    if (sessionCount > 0) {
      float avgSec = totalDuration / sessionCount;
      int minutes = (int)(avgSec / 60);
      int seconds = (int)(avgSec) % 60;
      int tenths = (int)(avgSec * 10) % 10;

      char buffer[9];
      snprintf(buffer, sizeof(buffer), "AVG%02d%02d%d", minutes, seconds, tenths);
      buffer[8] = buffer[7];
      buffer[7] = '.';
      buffer[9] = '\0';


      sevseg.setChars(buffer);
    } else {
      sevseg.setChars("--------");
    }
  }
}
