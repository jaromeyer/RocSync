# RocSync firmware

This folder contains the firmware. It uses platformio with the arduino core for CH32V003. Use their USB debugger to flash the PCB.


# How to build a PlatformIO based project

1. Install PlatformIO Core:
    Go to the [PlatformIO Website](https://docs.platformio.org/en/latest/core/installation/methods/installer-script.html#local-download-macos-linux-windows) and follow the instructions:

    - Download the `get-platformio.py` script (right click -> "save as")
    
    - in a command window got to where the file has beend downloaded to:
        ```shell
        cd <download folder>
        ```

    - run the script (installs PlatformIO core in a new vEnv, path is shown as a message)  
        ```shell
        python get-platformio.py
        ```

2. Activate the PlatformIO vEnv
    
    - the PlatformIO vEnv is usually created under `C:\users\<name>\.platformio\penv`. Navigate to the install location and find the `Scripts` folder.
        ```shell
        cd <...>\.platformio\penv\Scripts
        ```
    
    - activate the vEnv
        ```
        activate
        ```


3. Install support for the CH32V003 chip
    By default, PlatformIO doesn't offer support for the CH32V003 chip. An Arduino-style port of this chips functionality exists in [this repository](https://github.com/Community-PIO-CH32V/platform-ch32v). Follow its installation instructions:

    - the previously created vEnv needs to be activated in a terminal
    
    - install the CH32V003 library through PIO's own package manager: 
        ```shell
        pio pkg install -g -p https://github.com/Community-PIO-CH32V/platform-ch32v.git
        ```

4. Flashing the MC (with PlatformIO)
    To connect the CH32V003 to a computer, the [WCH-LinkE device](https://www.olimex.com/Products/RISC-V/WCH/WCH-LinkE/) is needed. Additionally, the [WCH-Utiliy](https://www.wch.cn/downloads/WCH-LinkUtility_ZIP.html) can be used to test the connection, configure the chip and manually flash binaries to the MC. However, to flash with PlatformIO, the Utility isn't strictly needed:

    - Connect the SWDIO and GND pin of the CH32V003 to the LinkE device (cables should ideally be < 30cm long).
    
    - make sure a battery is connected to the RocSync and it's switched on.

    - the previously created vEnv needs to be activated in a terminal.

    - navigate to a PIO project folder (where the firmware is located in a `src` folder)
        ```shell
        cd <project folder>
        ```
    - to compile and flash the code to the MC use the following command:
        ```shell
        pio run --target upload
        ```

