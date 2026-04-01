# RocSync Hardware
This folder contains the KiCad project for the RocSync device, designed with JLCPCB manufacturing and assembly in mind. Fabrication files were generated using the [JLCPCB tools plugin](https://github.com/Bouni/kicad-jlcpcb-tools). The latest production files are available [here](rev2/jlcpcb/production_files/).

## Accuracy
Revision 1 likely experienced accuracy issues documented [here](https://github.com/nlouman/documentation_todos_nino/blob/main/rocsync/1_checking_rocsync_clock.md), though never confirmed against a known accurate reference. Revision 2 replaced the crystal with a high-precision TCXO to improve absolute accuracy. Testing against a u-blox NEO-M8N GPS timepulse output showed ~1.3 ppm drift over 24 hours, roughly consistent with the TCXO's specification. This equates to 1 ms drift every 13 minutes. If required, one of the OLED pins could be used to feed a 1000 Hz timepulse signal from the GPS module into the microcontroller.

## Revision History
- **Rev1**: Initial design with 16-bit counter and crystal oscillator. Single-sided JLCPCB assembly with manual back-side soldering.
- **Rev2**: Improved design featuring a 20-bit counter, high-precision TCXO, 5-way button interface, and fifth corner LED for unambiguous orientation detection. Optimized for dual-sided assembly.

## Potential Improvements
- [ ] Decouple IR and visible LED brightness
- [ ] Address LED brightness degradation at low battery voltage (consider implementing a buck-boost converter)

## JLCPCB Ordering
1. Upload the **Gerber files** and let JLCPCB auto-populate all options.
2. Enable **PCB Assembly**, selecting **Standard PCBA** and **top side assembly**.
3. Upload the **BOM** and **POS files**.
4. Review the 3D preview and proceed to place your order.

## Mechanical BOM
| Part                                | Quantity | Source                                                              |
| ----------------------------------- | -------- | ------------------------------------------------------------------- |
| 604050 battery with PH2.0 connector | 1        | [Aliexpress](https://aliexpress.com/item/1005009787566737.html) or [Aliexpress](https://aliexpress.com/item/1005010639629430.html) |
| Handle                              | 1        | [3D printed](rev2/3d_models/handle.stl)                             |
| Cover                               | 1        | [3D printed](rev2/3d_models/top_cover.stl)                          |
| M3x10 screws                        | 4        | [Aliexpress](https://www.aliexpress.com/item/32850409234.html)      |
| M3 nuts                             | 4        | [Aliexpress](https://www.aliexpress.com/item/32977174437.html)      |
| Matte smoked film ~32x32 cm         | 1        | [Aliexpress](https://www.aliexpress.com/item/1005005636607163.html) |

## Assembly Instructions
> ⚠️ **Warning**: Verify battery polarity before turning on! The polarity of PH2.0 connectors may differ among battery manufacturers. The positive pin should be the one located closer to the center of the device. 

1. **Prepare the Film**  
   Cut a **30x30 cm square** from the matte smoked film. Place it with the **sticky side facing up** on a flat surface.

2. **Position the Faceplate**  
   Carefully center the **3D printed faceplate** on the film, ensuring the side with the **cutouts** that will later contact the PCB is facing up. If the film is curling, it can be helpful to have someone hold down the corners of the film while you position the faceplate.

3. **Create Center Flaps**  
   Use a cutter or knife to make an **X-shaped cut** in the center square hole of the film. Fold the resulting four flaps upwards and trim the tips, leaving a **2-3 cm strip**. Stick them to the faceplate, making sure to tension the film for a tight fit around the cutout.

4. **Clear Cutouts**  
   Cut away the film from the **cutouts** by slicing along their edges to detach the small film pieces. For the **top side** with **40 counter LEDs**, consider cutting the film into vertical strips between the LEDs and removing the sections covering the LEDs, rather than cutting out each hole individually.

5. **Attach the PCB**  
   Place the **PCB** facing down onto the prepared front plate, ensuring all LEDs are protruding through the cutouts. The PCB should fit snugly against the faceplate.

6. **Secure the PCB**  
    Cut away squares from the outer corners of the film to create flaps that can be folded around the PCB and adhered to its back. This will hold the PCB tightly against the faceplate. Make sure to firmly pull the film around the edge before sticking it down. It is easiest to start from the **middle** and work your way out to the corners, pulling the film tight each time before adhering a new section to the PCB.

7. **Attach the Handle**
   Use a cutter to poke holes through the film for the **four screws**. Attach the handle to the back of the PCB, ensuring the **battery cable** routes through the designated channel, and secure it with screws and nuts.

8. **Secure the Battery**  
   Adhere the battery to the back of the handle using **thin double-sided tape** to prevent it from falling out when the device is turned upside down. If the battery doesn't fit directly, you might have to fold its corners inwards to make it fit.

9.  **Connect the Battery and Power On**  
    Connect the battery to the PCB, ensuring the **correct polarity**. The positive pin should be the one closer to the center. When you turn on the power switch, the device should activate, and the LEDs will begin to blink.
