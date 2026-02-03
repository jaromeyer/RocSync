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
  // persistent counter, will just increment every time update is called (@2kHZ)
  static int ticks = 0;

  // generate clock signal
  digitalWrite(CLK_PIN, ticks % 2);

  if (ticks % 2 == 0)
  {
    // falling clock edge, do signal manipulation here
    // essentially writes a 1 every time the ring has circled fully, as the shift registers are chained open
    digitalWrite(RING_DATA_PIN, ticks % (PERIOD * 2) == 0);

    // this basically calculates 0.1s increments since the clock has turned on (ticks count up since start)
    int counter = (ticks - 1) / (PERIOD * 2) + 1;

    // the latch is activated exactly at the end of a period - when the counter should show the new value. Therefore, first 80 irrelevant bits have to be written, and then, the 20 counter bits at the end of the period. This is achieved by right-shifting the (binary) counter value by a wrap-around index (first the counter is shifted by 20-79 positions and produces "garbage" and then finally it is shifted by 0-19 positions to output the correct counter bits)
    int index = (ticks/2 + 19) % PERIOD;
    digitalWrite(COUNTER_DATA_PIN, counter >> index & 1);
  }

  // latch previously shifted counter into LED drivers (this finally displays the counter)
  digitalWrite(COUNTER_LATCH_PIN, ticks % (PERIOD * 2) == 1);


  ticks++;
}

void changeMode()
{

  digitalWrite(MODE_LED_PIN, HIGH);

  // static bool standby = false;
  // standby = !standby;
  // if (standby)
  // {
  //   // turn off LEDs
  //   analogWrite(RING_BLANK_PIN, 4096);
  //   analogWrite(COUNTER_BLANK_PIN, 4096);
  //   analogWrite(CORNER_ENABLE_PIN, 0);
  //   digitalWrite(MODE_LED_PIN, HIGH);
  // }
  // else
  // {
  //   // turn on LEDs with configured brightness
  //   analogWrite(RING_BLANK_PIN, (1 - RING_BRIGHTNESS/100.0) * 4096);
  //   analogWrite(COUNTER_BLANK_PIN, (1 - COUNTER_BRIGHTNESS/100.0) * 4096);
  //   analogWrite(CORNER_ENABLE_PIN, CORNER_BRIGHTNESS/100.0 * 4096);
  //   digitalWrite(MODE_LED_PIN, LOW);
  // }
}

static void SetSysClockTo_24MHz_HSE(void)
{
  __IO uint32_t StartUpCounter = 0, HSEStatus = 0;

  /* Close PA0-PA1 GPIO function */
  RCC->APB2PCENR |= RCC_AFIOEN;
  AFIO->PCFR1 |= (1 << 15);

  /* Enable BYPASS mode for HSE */
  RCC->CTLR |= ((uint32_t)RCC_HSEBYP);

  /* Enable HSE */
  RCC->CTLR |= ((uint32_t)RCC_HSEON);

  /* Wait till HSE is ready and if Time out is reached exit */
  do
  {
    HSEStatus = RCC->CTLR & RCC_HSERDY;
    StartUpCounter++;
  } while ((HSEStatus == 0) && (StartUpCounter != HSE_STARTUP_TIMEOUT));

  RCC->APB2PCENR |= RCC_AFIOEN;
  AFIO->PCFR1 |= (1 << 15);

  if ((RCC->CTLR & RCC_HSERDY) != RESET)
  {
    HSEStatus = (uint32_t)0x01;
  }
  else
  {
    HSEStatus = (uint32_t)0x00;
  }

  if (HSEStatus == (uint32_t)0x01)
  {
    /* Flash 0 wait state */
    FLASH->ACTLR &= (uint32_t)((uint32_t)~FLASH_ACTLR_LATENCY);
    FLASH->ACTLR |= (uint32_t)FLASH_ACTLR_LATENCY_0;

    /* HCLK = SYSCLK = APB1 */
    RCC->CFGR0 |= (uint32_t)RCC_HPRE_DIV1;

    /* Select HSE as system clock source */
    RCC->CFGR0 &= (uint32_t)((uint32_t)~(RCC_SW));
    RCC->CFGR0 |= (uint32_t)RCC_SW_HSE;
    /* Wait till HSE is used as system clock source */
    while ((RCC->CFGR0 & (uint32_t)RCC_SWS) != (uint32_t)0x04)
    {
    }
  }
  else
  {
    /*
     * If HSE fails to start-up, the application will have wrong clock
     * configuration. User can add here some code to deal with this error
     */
  }
}

void setup()
{

  // SetSysClockTo_24MHz_HSE(); // set system clock to 24 MHz using external oscillator in BYPASS mode

  // initialize IO
  pinMode(CLK_PIN, OUTPUT);
  pinMode(RING_DATA_PIN, OUTPUT);
  pinMode(RING_BLANK_PIN, OUTPUT);
  pinMode(RING_LATCH_PIN, OUTPUT);
  pinMode(COUNTER_DATA_PIN, OUTPUT);
  pinMode(COUNTER_BLANK_PIN, OUTPUT);
  pinMode(COUNTER_LATCH_PIN, OUTPUT);
  pinMode(CORNER_ENABLE_PIN, OUTPUT);
  pinMode(BUTTON_LADDER_PIN, INPUT_ANALOG);
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

  // configure Timer1 and attach IRQ
  HardwareTimer *timer1 = new HardwareTimer(TIM1);
  timer1->setOverflow(FREQ * 2, HERTZ_FORMAT);
  timer1->attachInterrupt(update);
  timer1->resume();
}

void loop() {
  // PROBLEM: when using SetSysClockTo_24MHz_HSE() the analogRead functionality seems to break. without HSE it works perfectly fine. 

  // handle buttons?
  int bttn_adc = analogRead(BUTTON_LADDER_PIN);

  if ((bttn_adc > 500) && (bttn_adc < 1024)) {
    digitalWrite(MODE_LED_PIN, HIGH);
  }
  else {
    digitalWrite(MODE_LED_PIN, LOW);
  }

  delay(100);

  


}

