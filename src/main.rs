use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{mpsc, Arc, Mutex};

fn main() -> Result<(), anyhow::Error> {
    // Initialize the default host for audio I/O
    let host = cpal::default_host();

    // Set up the microphone (input device)
    let input_device = host
        .default_input_device()
        .expect("No input device available");
    println!("Using input device: {}", input_device.name()?);

    // Set up the speakers (output device)
    let output_device = host
        .default_output_device()
        .expect("No output device available");
    println!("Using output device: {}", output_device.name()?);

    // Get the default input and output stream configurations
    let input_config = input_device
        .default_input_config()
        .expect("Failed to get default input config");
    let output_config = output_device
        .default_output_config()
        .expect("Failed to get default output config");

    println!("Input config: {:?}", input_config);
    println!("Output config: {:?}", output_config);

    // Create buffers to store the reference (speaker) and microphone signals
    let reference_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let microphone_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    // Clone the buffers for use in the input and output streams
    let reference_buffer_clone = Arc::clone(&reference_buffer);
    let microphone_buffer_clone = Arc::clone(&microphone_buffer);

    // Build the output stream (speakers)
    let output_stream = output_device.build_output_stream(
        &output_config.config(),
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            // Store the reference signal (audio being played through the speakers)
            let mut reference = reference_buffer_clone.lock().unwrap();
            reference.extend_from_slice(data);
            println!("Output stream callback: {} samples added", data.len()); // Debugging
        },
        move |err| {
            eprintln!("Output stream error: {}", err);
        },
        None, // No timeout
    )?;

    // Build the input stream (microphone)
    let input_stream = input_device.build_input_stream(
        &input_config.config(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // Store the microphone signal (audio captured by the microphone)
            let mut microphone = microphone_buffer_clone.lock().unwrap();
            microphone.extend_from_slice(data);
        },
        move |err| {
            eprintln!("Input stream error: {}", err);
        },
        None, // No timeout
    )?;

    // Start the input and output streams
    output_stream.play()?;
    input_stream.play()?;

    println!("Audio capture and playback are running. Press Ctrl+C to stop.");

    // Keep the program running to allow audio processing
    loop {
        // Periodically print the buffer sizes for debugging
        let reference = reference_buffer.lock().unwrap();
        let microphone = microphone_buffer.lock().unwrap();
        println!(
            "Reference buffer size: {}, Microphone buffer size: {}",
            reference.len(),
            microphone.len(),
        );
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
