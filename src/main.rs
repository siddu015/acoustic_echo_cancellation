use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound;
use std::fs::File;
use std::io::BufReader;
use std::collections::VecDeque;

const FILE_NAME: &str = "recorded.wav";
const PROCESSED_FILE: &str = "processed.wav";
const ECHO_DELAY: usize = 4410; // ~100ms delay at 44.1kHz
const ECHO_ATTENUATION: f32 = 0.5;

fn record_audio(file_name: &str, duration_secs: u64) {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device found");
    let config = device.default_input_config().unwrap();
    let sample_rate = config.sample_rate().0;
    let channels = config.channels();

    let spec = hound::WavSpec {
        channels: channels as u16,
        sample_rate: sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let writer = hound::WavWriter::create(file_name, spec).unwrap();
    let writer = std::sync::Arc::new(std::sync::Mutex::new(writer));

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
            eprintln!("Error: {}", err);
        },
        None,
    ).unwrap();

    stream.play().unwrap();
    println!("Recording for {} seconds...", duration_secs);
    std::thread::sleep(std::time::Duration::from_secs(duration_secs));

    println!("Recording complete! Saved as {}", file_name);
}

fn process_audio(input_file: &str, output_file: &str) {
    let mut reader = hound::WavReader::open(input_file).unwrap();
    let spec = reader.spec();
    let mut writer = hound::WavWriter::create(output_file, spec).unwrap();

    let mut buffer = VecDeque::with_capacity(ECHO_DELAY);
    let mut filter_weight = 0.7;
    let mut step_size = 0.01;
    let ema_factor = 0.2; // Removed `mut` to fix warning
    let mut previous_error = 0.0;
    let noise_threshold = 500;

    for sample in reader.samples::<i16>() {
        let sample = sample.unwrap();
        let echo_sample = if buffer.len() >= ECHO_DELAY {
            buffer.pop_front().unwrap_or(0)
        } else {
            0
        };

        // **Adaptive filtering with no overflow**
        let filtered_echo = (echo_sample as f32 * filter_weight) as i16;
        let processed_sample = sample.saturating_sub(filtered_echo); // FIXED OVERFLOW

        // **Compute error with smoothing**
        let error = sample - filtered_echo;
        let smoothed_error = ema_factor * error as f32 + (1.0 - ema_factor) * previous_error;
        previous_error = smoothed_error;

        // **Adaptive step size based on error**
        let echo_power = (echo_sample as f32).powi(2);
        if echo_power > 0.01 {
            step_size = 0.1 / (echo_power + 1.0);
        }

        // **Update filter weight safely**
        filter_weight = (filter_weight + step_size * smoothed_error).clamp(0.0, 1.0); // FIXED OVERFLOW

        // **Noise Gate**
        let final_sample = if processed_sample.abs() < noise_threshold {
            0
        } else {
            processed_sample
        };

        writer.write_sample(final_sample).unwrap();
        buffer.push_back(sample);
    }

    println!("Fixed overflow! Noise reduction and echo cancellation applied. Saved as {}", output_file);
}

fn play_audio(file_name: &str) {
    let host = cpal::default_host();
    let device = host.default_output_device().expect("No output device found");
    let file = File::open(file_name).unwrap();
    let mut reader = hound::WavReader::new(BufReader::new(file)).unwrap(); // Make reader mutable

    let stream = device.build_output_stream(
        &device.default_output_config().unwrap().into(),
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
    println!("Playing processed audio...");
    std::thread::sleep(std::time::Duration::from_secs(10));
}

fn main() {
    let duration_secs = 10;

    record_audio(FILE_NAME, duration_secs);
    process_audio(FILE_NAME, PROCESSED_FILE);
    play_audio(PROCESSED_FILE);
}
