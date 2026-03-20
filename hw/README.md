# RocSync Hardware
This folder contains the KiCad project for the RocSync device, designed with JLCPCB manufacturing and assembly in mind. Fabrication files were generated using the [JLCPCB tools plugin](https://github.com/Bouni/kicad-jlcpcb-tools). The latest production files are available [here](rev2/jlcpcb/production_files/).

## Revision History
- **Rev1**: Initial design with 16-bit counter and crystal oscillator. Single-sided JLCPCB assembly with manual back-side soldering.
- **Rev2**: Improved design featuring a 20-bit counter, high-precision TCXO, 5-way button interface, and fifth corner LED for unambiguous orientation detection. Optimized for dual-sided assembly.

## TODO
- [ ] Decouple IR and visible LED brightness
- [ ] Address decreasing brightness at low battery voltage (consider buck-boost converter)

## Mechanical BOM
| Part                        | Quantity | Source                                                              |
| --------------------------- | -------- | ------------------------------------------------------------------- |
| 604050 battery              | 1        | [Aliexpress](https://www.aliexpress.com/item/1005007605225540.html) |
| Handle                      | 1        | [3D printed](rev2/3d_models/handle.stl)                             |
| Cover                       | 1        | [3D printed](rev2/3d_models/top_cover.stl)                          |
| M3x10 screws                | 4        | [Aliexpress](https://www.aliexpress.com/item/32850409234.html)      |
| M3 nuts                     | 4        | [Aliexpress](https://www.aliexpress.com/item/32977174437.html)      |
| Matte smoked film ~32x32 cm | 1        | [Aliexpress](https://www.aliexpress.com/item/1005005636607163.html) |

## Assembly Instructions
*TODO*
