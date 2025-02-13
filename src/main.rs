use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound;
use std::sync::{Arc, Mutex};

fn main() {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device found");
    println!("Using input device: {}", device.name().unwrap());

    let config = device.default_input_config().unwrap();
    println!("Input format: {:?}", config);

    let sample_rate = config.sample_rate().0;
    let channels = config.channels();

    // Create a WAV file writer
    let spec = hound::WavSpec {
        channels: channels as u16,
        sample_rate: sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let writer = Arc::new(Mutex::new(hound::WavWriter::create("output.wav", spec).unwrap()));

    // Create and start the audio stream
    let writer_clone = writer.clone();
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut writer = writer_clone.lock().unwrap();
            for &sample in data {
                let amplitude = (sample * i16::MAX as f32) as i16;
                writer.write_sample(amplitude).unwrap();
            }
        },
        move |err| {
            eprintln!("Error occurred: {}", err);
        },
        None,
    ).unwrap();

    stream.play().unwrap();

    println!("Recording... Speak into the microphone! Press Ctrl+C to stop.");
    std::thread::sleep(std::time::Duration::from_secs(10)); // Record for 10 seconds

    println!("Recording complete! Saved as output.wav");
}
