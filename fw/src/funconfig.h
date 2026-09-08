#ifndef _FUNCONFIG_H
#define _FUNCONFIG_H

// Place configuration items here, you can see a full list in ch32fun/ch32fun.h
// To reconfigure to a different processor, update TARGET_MCU in the  Makefile
#define FUNCONF_USE_HSE 1    // enable external 24MHz oscillator on PA1 PA2
#define FUNCONF_USE_HSI 0    // disable internal 24MHz oscillator
#define FUNCONF_USE_PLL 0    // disable PLL x2 for 24MHz system clock
#define FUNCONF_HSE_BYPASS 0 // use HSE circuit (amplifier?) despite external oscillator (not crystal) since it outputs clipped sine-wave instead of TTL levels

#define FUNCONF_SYSTICK_USE_HCLK 1

#endif
