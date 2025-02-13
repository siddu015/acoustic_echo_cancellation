use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound;
use std::sync::{Arc, Mutex};
use std::fs::File;
use std::io::BufReader;

fn record_audio(file_name: &str, duration_secs: u64) {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device found");
    println!("Using input device: {}", device.name().unwrap());

    let config = device.default_input_config().unwrap();
    println!("Input format: {:?}", config);

    let sample_rate = config.sample_rate().0;
    let channels = config.channels();

    let spec = hound::WavSpec {
        channels: channels as u16,
        sample_rate: sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let writer = Arc::new(Mutex::new(hound::WavWriter::create(file_name, spec).unwrap()));

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
    std::thread::sleep(std::time::Duration::from_secs(duration_secs));

    println!("Recording complete! Saved as {}", file_name);
}

fn play_audio(file_name: &str) {
    let host = cpal::default_host();
    let device = host.default_output_device().expect("No output device found");
    println!("Using output device: {}", device.name().unwrap());

    let file = File::open(file_name).unwrap();
    let mut reader = hound::WavReader::new(BufReader::new(file)).unwrap();

    let config = device.default_output_config().unwrap();

    let stream = device.build_output_stream(
        &config.into(),
        move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
            for (sample, out) in reader.samples::<i16>().zip(output.iter_mut()) {
                *out = sample.unwrap() as f32 / i16::MAX as f32;
            }
        },
        move |err| {
            eprintln!("Playback error: {}", err);
        },
        None,
    ).unwrap();

    stream.play().unwrap();
    println!("Playing audio...");

    std::thread::sleep(std::time::Duration::from_secs(5)); // Adjust as needed
}

fn main() {
    let file_name = "output.wav";
    let duration_secs = 5;

    record_audio(file_name, duration_secs);
    play_audio(file_name);
}
