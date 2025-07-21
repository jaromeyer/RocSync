#include <Arduino.h>

// pin definitions
const int CLK_PIN = PD0;
const int RING_DATA_PIN = PD2;
const int RING_BLANK_PIN = PD3;
const int RING_LATCH_PIN = PC1;
const int COUNTER_DATA_PIN = PD5;
const int COUNTER_BLANK_PIN = PD4;
const int COUNTER_LATCH_PIN = PD6;
const int CORNER_ENABLE_PIN = PC0;
const int MODE_BUTTON_PIN = PC2;
const int MODE_LED_PIN = PC3;

// config
const int FREQ = 1000; // in hz
const int PERIOD = 100;
const int RING_BRIGHTNESS = 100; // 0-100%
const int COUNTER_BRIGHTNESS = 50;
const int CORNER_BRIGHTNESS = 100;

void update()
{
  static int ticks = 0;

  // generate clock signal
  digitalWrite(CLK_PIN, ticks % 2);

  if (ticks % 2 == 0)
  {
    // falling clock edge, do signal manipulation here
    digitalWrite(RING_DATA_PIN, ticks % (PERIOD * 2) == 0);
    int counter = (ticks - 1) / (PERIOD * 2) + 1;
    int index = (ticks / 2 + 15) % PERIOD;
    digitalWrite(COUNTER_DATA_PIN, counter >> index & 1);
  }

  // latch previously shifted counter into LED drivers
  digitalWrite(COUNTER_LATCH_PIN, ticks % (PERIOD * 2) == 1);

  ticks++;
}

void changeMode()
{
  static bool standby = false;
  standby = !standby;
  if (standby)
  {
    // turn off LEDs
    analogWrite(RING_BLANK_PIN, 4096);
    analogWrite(COUNTER_BLANK_PIN, 4096);
    analogWrite(CORNER_ENABLE_PIN, 0);
    digitalWrite(MODE_LED_PIN, HIGH);
  }
  else
  {
    // turn on LEDs with configured brightness
    analogWrite(RING_BLANK_PIN, (1 - RING_BRIGHTNESS / 100.0) * 4096);
    analogWrite(COUNTER_BLANK_PIN, (1 - COUNTER_BRIGHTNESS / 100.0) * 4096);
    analogWrite(CORNER_ENABLE_PIN, CORNER_BRIGHTNESS / 100.0 * 4096);
    digitalWrite(MODE_LED_PIN, LOW);
  }
}

void setup()
{
  // initialize IO
  pinMode(CLK_PIN, OUTPUT);
  pinMode(RING_DATA_PIN, OUTPUT);
  pinMode(RING_BLANK_PIN, OUTPUT);
  pinMode(RING_LATCH_PIN, OUTPUT);
  pinMode(COUNTER_DATA_PIN, OUTPUT);
  pinMode(COUNTER_BLANK_PIN, OUTPUT);
  pinMode(COUNTER_LATCH_PIN, OUTPUT);
  pinMode(CORNER_ENABLE_PIN, OUTPUT);
  pinMode(MODE_BUTTON_PIN, INPUT_PULLUP);
  pinMode(MODE_LED_PIN, OUTPUT);

  // set outputs to initial values
  digitalWrite(RING_DATA_PIN, LOW);
  digitalWrite(RING_BLANK_PIN, HIGH);
  digitalWrite(RING_LATCH_PIN, HIGH);
  digitalWrite(COUNTER_DATA_PIN, LOW);
  digitalWrite(COUNTER_BLANK_PIN, HIGH);
  digitalWrite(COUNTER_LATCH_PIN, HIGH);
  digitalWrite(MODE_LED_PIN, LOW);

  // set brightness using PWM
  analogWriteFrequency(8000); // set PWM frequency (internally sets frequency for Timer2)
  analogWrite(RING_BLANK_PIN, (1 - RING_BRIGHTNESS / 100.0) * 4096);
  analogWrite(COUNTER_BLANK_PIN, (1 - COUNTER_BRIGHTNESS / 100.0) * 4096);
  analogWrite(CORNER_ENABLE_PIN, CORNER_BRIGHTNESS / 100.0 * 4096);

  // clear shift registers
  for (int i = 0; i < PERIOD; i++)
  {
    digitalWrite(CLK_PIN, LOW);
    delayMicroseconds(10);
    digitalWrite(CLK_PIN, HIGH);
    delayMicroseconds(10);
  }

  // attach button IRQ
  attachInterrupt(MODE_BUTTON_PIN, GPIO_Mode_IPU, &changeMode, EXTI_Mode_Interrupt, EXTI_Trigger_Falling);

  // configure Timer1 and attach IRQ
  HardwareTimer *timer1 = new HardwareTimer(TIM1);
  timer1->setOverflow(FREQ * 2, HERTZ_FORMAT);
  timer1->attachInterrupt(update);
  timer1->resume();
}

void loop() {}
