use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::mpsc;

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

    // Create a channel to send audio data from the input stream to the output stream
    let (tx, rx) = mpsc::sync_channel(1024); // Use a sync channel with a buffer size

    // Build the input stream (microphone)
    let input_stream = input_device.build_input_stream(
        &input_config.config(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // Send the captured audio data to the output stream
            if let Err(err) = tx.send(data.to_vec()) {
                eprintln!("Error sending audio data: {}", err);
            }
        },
        move |err| {
            eprintln!("Input stream error: {}", err);
        },
        None, // No timeout
    )?;

    // Build the output stream (speakers)
    let output_stream = output_device.build_output_stream(
        &output_config.config(),
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            // Receive audio data from the input stream and play it back
            if let Ok(input_data) = rx.try_recv() {
                for (output_sample, input_sample) in data.iter_mut().zip(input_data.iter()) {
                    *output_sample = *input_sample;
                }
            }
        },
        move |err| {
            eprintln!("Output stream error: {}", err);
        },
        None, // No timeout
    )?;

    // Start the input and output streams
    input_stream.play()?;
    output_stream.play()?;

    println!("Audio loopback is running. Press Ctrl+C to stop.");

    // Keep the program running to allow audio processing
    loop {}
}
