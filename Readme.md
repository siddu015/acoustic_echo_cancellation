# Acoustic Echo Cancellation in Rust

This project demonstrates how to build a **Real-Time Acoustic Echo Cancellation (AEC) System** using Rust. The system captures audio from your microphone, processes it to remove echoes, and plays back the cleaned audio. It’s a great starting point for learning about audio processing and adaptive filtering in Rust.

---

## **How It Works**

1. **Recording**: The program captures audio from your microphone using the `cpal` crate and saves it as a `.wav` file.
2. **Processing**: The recorded audio is processed using an **adaptive filter** to estimate and remove echoes.
3. **Playback**: The cleaned audio is played back using your default audio output device.

---

## **Getting Started**

### **Prerequisites**

- Install Rust from [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install).
- Verify the installation:
  ```bash
  rustc --version
  ```

### **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/siddu015/acoustic_echo_cancellation.git
   cd acoustic_echo_cancellation
   ```

2. Build the project:
   ```bash
   cargo build
   ```

---

## **Usage**

1. **Record Audio**:
    - Run the program to record audio for 10 seconds:
      ```bash
      cargo run
      ```
    - The recorded audio will be saved as `recorded.wav`.

2. **Process Audio**:
    - The program will automatically process the recorded audio to remove echoes.
    - The cleaned audio will be saved as `processed.wav`.

3. **Play Audio**:
    - The program will play back the processed audio.

---

## **How to Modify**

- **Change Recording Duration**: Modify the `duration_secs` variable in `main()`.
- **Adjust Filter Parameters**: Tweak `filter_len` and `mu` in the `process_audio` function.
- **Add Noise Reduction**: Extend the `process_audio` function to include noise reduction.

---

## **Example Output**

1. **Recording**:
   ```
   Recording for 10 seconds...
   Recording complete! Saved as recorded.wav
   ```

2. **Processing**:
   ```
   Adaptive echo cancellation applied! Saved as processed.wav
   ```

3. **Playback**:
   ```
   Playing processed audio...
   ```

---

## **Contributing**

Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.


---

Enjoy building and experimenting with audio processing in Rust! 🚀
