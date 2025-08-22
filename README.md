## CS5272 Embedded Systems Project Repository

This repository contains the project files for our **CS5272 Embedded Systems** course project. You can find the report [here](./Report.pdf)

### 📂 Folder Descriptions

- **`hardware/`**
  Files related to running inference on artemis hardware, motor control and hardware stls. 
  - **`mcu_inference/`**  
    Code for deploying inference code on micro-controller and latency and memory measurement with static data files. Simply upload the `.ino` file here to the board.
  - **`mcu_live_inference/`**  
    Code for deploying inference code on micro-controller with live audio and vibration data capture. Simply upload the `.ino` file here to the board.
  - **`motor_controller/`**  
    Code for the  motor controller driving the motor responsible for rotating the rover wheel across different surfaces.
  - **`wheel-stls/`**  
    3D STL design files for the rover wheels used in surface traversal.

- **`data-collection/`**  
  Scripts and code for recording audio and converting it into the desired format.
  - **`data-vibration/`**  
    Contains recorded vibrational signals collected over various surfaces.
  - **`data/`**  
    Contains recorded audio signals captured across different surfaces.

- **`hdc/`**  
  Code for training (on a regular computer) and inference (on a microcontroller) using the **Hyperdimensional Computing (HDC)** paradigm on both audio and vibration sensor data.

- **`misc/`**
  Other files used for our project.
  - **`sensor_tests/`**  
    Test scripts for various sensors used in the project, including analog mics, digital mics, and vibration sensors.
  - **`demo/`**  
    Photos and videos of the experimental setup and demonstration materials.
  - **`meas/`**  
    Images and CSV files for power profiles using Nordic power profiler.
