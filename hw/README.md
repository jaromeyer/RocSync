# RocSync hardware
This folder contains the KiCad project for the RocSync device. It was designed with manufacturing and assembly by JLCPCB in mind. The following [plugin](https://github.com/Bouni/kicad-jlcpcb-tools) was used to generate fabrication files for JLCPCB. The folder `3d_models` contains the STEP and STL files for the cover and handle.

## TODO
- Use TCXO for better accuracy (e.g. [TX0283D](https://jlcpcb.com/partdetail/Tst-TX0283D/C499191))
- Decouple IR and visible LED brightness
- Fix decreasing brightness when battery voltage is low
- Verify that handle fits the battery

## Mechanical BOM
| Part                        | Quantity | Source                                                              |
| --------------------------- | -------- | ------------------------------------------------------------------- |
| 604050 battery              | 1        | [Aliexpress](https://www.aliexpress.com/item/1005007605225540.html) |
| Handle                      | 1        | 3d printed                                                          |
| Cover                       | 1        | 3d printed                                                          |
| M3x10 screws                | 4        | [Aliexpress](https://www.aliexpress.com/item/32850409234.html)      |
| M3 nuts                     | 4        | [Aliexpress](https://www.aliexpress.com/item/32977174437.html)      |
| matte smoked film ~32x32 cm | 1        | [Aliexpress](https://www.aliexpress.com/item/1005005636607163.html) |