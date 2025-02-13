use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat};

fn main() {
    let host = cpal::default_host();

    // Get the default input device (microphone)
    let device = host.default_input_device().expect("No input device found");
    println!("Using input device: {}", device.name().unwrap());

    // Get the input device configuration
    let config = device.default_input_config().unwrap();
    println!("Input format: {:?}", config);

    // Create and start the audio stream
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // Print the captured audio samples
            println!("{:?}", &data[0..10]); // Print first 10 samples
        },
        move |err| {
            eprintln!("Error occurred: {}", err);
        },
        None,
    ).unwrap();

    // Keep the stream alive
    stream.play().unwrap();

    println!("Listening... Press Ctrl+C to stop.");
    std::thread::sleep(std::time::Duration::from_secs(10)); // Capture for 10 seconds
}
