use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound;
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::Arc;
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

    let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    let mut buffer: Vec<f32> = samples.iter().map(|&s| s as f32).collect();

    let fft_size = 1024;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    let mut complex_buffer: Vec<Complex<f32>> = vec![Complex { re: 0.0, im: 0.0 }; fft_size];

    for chunk in buffer.chunks_mut(fft_size) {
        let len = chunk.len();
        for i in 0..len {
            complex_buffer[i] = Complex {
                re: chunk[i],
                im: 0.0,
            };
        }

        // Apply FFT
        fft.process(&mut complex_buffer);

        // Apply Echo Cancellation in Frequency Domain
        for bin in &mut complex_buffer {
            let magnitude = bin.norm();
            if magnitude > 500.0 {
                *bin *= 0.5; // Reduce echo
            }
        }

        // Apply Inverse FFT
        ifft.process(&mut complex_buffer);

        // Convert back to time domain
        for i in 0..len {
            chunk[i] = complex_buffer[i].re;
        }
    }

    // Save the processed audio
    for &sample in &buffer {
        writer.write_sample(sample as i16).unwrap();
    }

    println!("Echo reduced using FFT! Saved as {}", output_file);
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
