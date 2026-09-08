#include "ch32fun.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Config
#define FREQ 1000
#define REV2 // "REV1" or "REV2"

// HW specific definitions
#if defined(REV1)
#define PIN_CLK PD0
#define PIN_RING_DATA PD2
#define PIN_RING_BLANK PD3 // T2CH2
#define PIN_RING_LATCH PC1
#define PIN_COUNTER_DATA PD5
#define PIN_COUNTER_BLANK PD4 // T2CH1ETR
#define PIN_COUNTER_LATCH PD6
#define PIN_CORNER_ENABLE PC0 // T2CH3
#define PIN_BUTTON PC2
#define PIN_MODE_LED PC3
#define COUNTER_BITS 16
#define PERIOD 100
#elif defined(REV2)
#define PIN_CLK PD0
#define PIN_RING_DATA PD2
#define PIN_RING_BLANK PD3 // T2CH2
#define PIN_RING_LATCH PC7
#define PIN_COUNTER_DATA PD5
#define PIN_COUNTER_BLANK PD4 // T2CH1ETR
#define PIN_COUNTER_LATCH PD6
#define PIN_CORNER_ENABLE PC0 // T2CH3
#define PIN_BUTTON PC4        // A2
#define PIN_MODE_LED PC3
#define COUNTER_BITS 20
#define PERIOD 100
#endif

int ring_brightness = 100;
int counter_brightness = 50;
int corner_brightness = 100;

bool standby = false;

void SysTick_Handler(void) __attribute__((interrupt));
void SysTick_Handler(void)
{
    // Reference: https://github.com/cnlohr/ch32fun/tree/master/examples/systick_irq

    static uint32_t ticks = 0;

    // Increment the Compare Register for the next trigger
    // If more than this number of ticks elapse before the trigger is reset,
    // you may miss your next interrupt trigger
    // (Make sure thae IQR is lightweight and CMP value is reasonble)
    SysTick->CMP += 500000 / FREQ * DELAY_US_TIME;

    // generate clock signal: ticks % 2
    funDigitalWrite(PIN_CLK, (ticks % 2));

    if ((ticks % 2) == 0)
    {
        // falling clock edge logic
        // RING_DATA: ticks % (PERIOD * 2) == 0
        funDigitalWrite(PIN_RING_DATA, (ticks % (PERIOD * 2) == 0));

        // Counter logic
        int counter = (ticks - 1) / (PERIOD * 2) + 1;
        int index = (ticks / 2 + COUNTER_BITS - 1) % PERIOD;
        funDigitalWrite(PIN_COUNTER_DATA, (counter >> index) & 1);
    }

    // COUNTER_LATCH: ticks % (PERIOD * 2) == 1
    funDigitalWrite(PIN_COUNTER_LATCH, (ticks % (PERIOD * 2) == 1));

    // Clear the trigger state for the next IRQ
    SysTick->SR = 0x00000000;

    ticks++;
}

void set_counter(int counter)
{
    for (int i = 0; i < COUNTER_BITS; i++)
    {
        funDigitalWrite(PIN_COUNTER_DATA, (counter >> i) & 1);
        funDigitalWrite(PIN_CLK, FUN_LOW);
        Delay_Us(1);
        funDigitalWrite(PIN_CLK, FUN_HIGH);
        Delay_Us(1);
    }
}

void setup_gpio()
{
    // Enable all GPIO ports
    funGpioInitAll();

    // Setup logic outputs (Push-Pull)
    funPinMode(PIN_CLK, GPIO_Speed_10MHz | GPIO_CNF_OUT_PP);
    funPinMode(PIN_RING_DATA, GPIO_Speed_10MHz | GPIO_CNF_OUT_PP);
    funPinMode(PIN_RING_LATCH, GPIO_Speed_10MHz | GPIO_CNF_OUT_PP);
    funPinMode(PIN_COUNTER_DATA, GPIO_Speed_10MHz | GPIO_CNF_OUT_PP);
    funPinMode(PIN_COUNTER_LATCH, GPIO_Speed_10MHz | GPIO_CNF_OUT_PP);
    funPinMode(PIN_MODE_LED, GPIO_Speed_10MHz | GPIO_CNF_OUT_PP);

    // Setup brigthness PWM outputs (Alternate Function Push-Pull)
    funPinMode(PIN_RING_BLANK, GPIO_Speed_10MHz | GPIO_CNF_OUT_PP_AF);
    funPinMode(PIN_COUNTER_BLANK, GPIO_Speed_10MHz | GPIO_CNF_OUT_PP_AF);
    funPinMode(PIN_CORNER_ENABLE, GPIO_Speed_10MHz | GPIO_CNF_OUT_PP_AF);

// Setup button input
#if defined(REV1)
    // Setup as input with pull-up
    funPinMode(PIN_BUTTON, GPIO_CFGLR_IN_PUPD);
    funDigitalWrite(PIN_BUTTON, FUN_HIGH);
#elif defined(REV2)
    // Enable ADC
    funAnalogInit();
    funPinMode(PIN_BUTTON, GPIO_CFGLR_IN_ANALOG);
#endif
}

void systick_init(void)
{
    // Reference: https://github.com/cnlohr/ch32fun/tree/master/examples/systick_irq

    // Reset any pre-existing configuration
    SysTick->CTLR = 0x0000;

    // Set the compare register to trigger every 500us (2kHz)
    SysTick->CMP = 500000 / FREQ * DELAY_US_TIME - 1;

    // Reset the Count Register to 0
    SysTick->CNT = 0x00000000;

    // Set the SysTick Configuration
    // NOTE: By not setting SYSTICK_CTLR_STRE, we maintain compatibility with
    // busywait delay funtions used by ch32v003_fun.
    SysTick->CTLR |= SYSTICK_CTLR_STE |  // Enable Counter
                     SYSTICK_CTLR_STIE | // Enable Interrupts
                     SYSTICK_CTLR_STCLK; // Set Clock Source to HCLK/1

    // Enable the SysTick IRQ
    NVIC_EnableIRQ(SysTick_IRQn);
}

void t2pwm_init(void)
{
    // Reference: https://github.com/cnlohr/ch32fun/tree/master/examples/tim2_pwm

    // Enable TIM2
    RCC->APB1PCENR |= RCC_APB1Periph_TIM2;

    // Reset TIM2 to init all regs
    RCC->APB1PRSTR |= RCC_APB1Periph_TIM2;
    RCC->APB1PRSTR &= ~RCC_APB1Periph_TIM2;

    // SMCFGR: default clk input is CK_INT
    // set TIM2 clock prescaler divider
    TIM2->PSC = 0x0000;
    // set PWM total cycle width
    TIM2->ATRLR = 255;

    // for channel 1 and 2, let CCxS stay 00 (output), set OCxM to 110 (PWM I)
    // enabling preload causes the new pulse width in compare capture register only to come into effect when UG bit in SWEVGR is set (= initiate update) (auto-clears)
    TIM2->CHCTLR1 |= TIM_OC1M_2 | TIM_OC1M_1 | TIM_OC1PE | TIM_OC2M_2 | TIM_OC2M_1 | TIM_OC2PE;
    TIM2->CHCTLR2 |= TIM_OC3M_2 | TIM_OC3M_1 | TIM_OC3PE;

    // CTLR1: default is up, events generated, edge align
    // enable auto-reload of preload
    TIM2->CTLR1 |= TIM_ARPE;

    // Enable Channel outputs, set default state
    // ring and counter are active low, corner is active high
    TIM2->CCER |= TIM_CC1E | (TIM_CC1P & 0xFFFF);
    TIM2->CCER |= TIM_CC2E | (TIM_CC2P & 0xFFFF);
    TIM2->CCER |= TIM_CC3E | (TIM_CC3P & 0x0000);

    // initialize counter
    TIM2->SWEVGR |= TIM_UG;

    // Start TIM2
    TIM2->CTLR1 |= TIM_CEN;
}

void set_counter_brightness(int brightness)
{
    TIM2->CH1CVR = brightness * 255 / 100;
}

void set_ring_brightness(int brightness)
{
    TIM2->CH2CVR = brightness * 255 / 100;
}

void set_corner_brightness(int brightness)
{
    TIM2->CH3CVR = brightness * 255 / 100;
}

void verify_hse()
{
    // Check if HSE is ready and used as system clock
    // If not, blink corner LEDs to indicate error
    if ((RCC->CTLR & RCC_HSERDY) == 0 || (RCC->CFGR0 & RCC_SWS) != RCC_SWS_HSE)
    {
        while (1)
        {
            funDigitalWrite(PIN_CORNER_ENABLE, FUN_HIGH);
            Delay_Ms(100);
            funDigitalWrite(PIN_CORNER_ENABLE, FUN_LOW);
            Delay_Ms(100);
        }
    }
}

void poll_buttons()
{
    static int last_bttn_index = -1;
    int bttn_index = -1;

#if defined(REV1)
    if (funDigitalRead(PIN_BUTTON) == FUN_LOW)
    {
        bttn_index = 1; // middle button
    }
#elif defined(REV2)
    int bttn_adc = 1024 - funAnalogRead(ANALOG_2);

    // Round ADC reading to nearest button index (1-5) or 0 for no button
    int nearest_index = (bttn_adc * 5 + 512) / 1024;
    if (nearest_index > 0 && abs(nearest_index * 1024 / 5 - bttn_adc) < 50)
    {
        bttn_index = nearest_index - 1;
    }
#endif

    if (bttn_index == last_bttn_index)
        return;

    switch (bttn_index)
    {
    case 0: // up
        counter_brightness = (counter_brightness + 10);
        if (counter_brightness > 100)
            counter_brightness = 100;
        if (!standby)
            set_counter_brightness(counter_brightness);
        break;
    case 1: // middle
        standby = !standby;
        if (standby)
        {
            set_corner_brightness(0);
            set_counter_brightness(0);
            set_ring_brightness(0);
            funDigitalWrite(PIN_MODE_LED, FUN_HIGH);
        }
        else
        {
            set_corner_brightness(corner_brightness);
            set_counter_brightness(counter_brightness);
            set_ring_brightness(ring_brightness);
            funDigitalWrite(PIN_MODE_LED, FUN_LOW);
        }
        break;
    case 2: // down
        counter_brightness = (counter_brightness - 10);
        if (counter_brightness < 0)
            counter_brightness = 0;
        if (!standby)
            set_counter_brightness(counter_brightness);
        break;
    case 3: // right
        ring_brightness = (ring_brightness + 10);
        if (ring_brightness > 100)
            ring_brightness = 100;
        if (!standby)
            set_ring_brightness(ring_brightness);
        break;
    case 4: // left
        ring_brightness = (ring_brightness - 10);
        if (ring_brightness < 0)
            ring_brightness = 0;
        if (!standby)
            set_ring_brightness(ring_brightness);
        break;
    }
    last_bttn_index = bttn_index;
}

int main()
{
    SystemInit();
    setup_gpio();
    verify_hse();
    t2pwm_init();

    // Initial states
    funDigitalWrite(PIN_RING_DATA, FUN_LOW);
    funDigitalWrite(PIN_RING_LATCH, FUN_HIGH);
    funDigitalWrite(PIN_COUNTER_DATA, FUN_LOW);
    funDigitalWrite(PIN_COUNTER_LATCH, FUN_HIGH);
    funDigitalWrite(PIN_MODE_LED, FUN_LOW);
    set_corner_brightness(0);
    set_counter_brightness(0);
    set_ring_brightness(0);

    // Clear shift registers manually
    for (int i = 0; i < PERIOD; i++)
    {
        funDigitalWrite(PIN_CLK, FUN_LOW);
        Delay_Us(10);
        funDigitalWrite(PIN_CLK, FUN_HIGH);
        Delay_Us(10);
    }

    set_corner_brightness(corner_brightness);
    set_counter_brightness(counter_brightness);
    set_ring_brightness(ring_brightness);

    // Start clock
    systick_init();

    while (1)
    {
        poll_buttons();
        Delay_Ms(50);
    }
}
