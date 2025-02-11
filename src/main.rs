use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

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
    let cleaned_audio_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    // Clone the buffers for use in the input and output streams
    let reference_buffer_clone = Arc::clone(&reference_buffer);
    let microphone_buffer_clone = Arc::clone(&microphone_buffer);
    let cleaned_audio_buffer_clone = Arc::clone(&cleaned_audio_buffer);

    // Initialize the adaptive filter (NLMS)
    let filter_length = 256; // Length of the adaptive filter
    let mut filter = NlmsFilter::new(filter_length, 0.1); // Step size = 0.1

    // Build the output stream (speakers)
    let output_stream = output_device.build_output_stream(
        &output_config.config(),
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            // Play the cleaned audio from the cleaned_audio_buffer
            let mut cleaned_audio = cleaned_audio_buffer_clone.lock().unwrap();
            let len = data.len().min(cleaned_audio.len());
            data[..len].copy_from_slice(&cleaned_audio[..len]);
            cleaned_audio.drain(..len);
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

            // Perform echo cancellation
            let mut reference = reference_buffer.lock().unwrap();
            if reference.len() >= filter_length && microphone.len() >= filter_length {
                let echo_cancelled = filter.process(&reference, &microphone);

                // Store the cleaned audio in the cleaned_audio_buffer
                let mut cleaned_audio = cleaned_audio_buffer.lock().unwrap();
                cleaned_audio.extend_from_slice(&echo_cancelled);

                // Clear the buffers to avoid overflow
                reference.drain(..filter_length);
                microphone.drain(..filter_length);
            }
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
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

// NLMS Adaptive Filter Implementation
struct NlmsFilter {
    filter_length: usize,
    step_size: f32,
    weights: Vec<f32>,
}

impl NlmsFilter {
    fn new(filter_length: usize, step_size: f32) -> Self {
        Self {
            filter_length,
            step_size,
            weights: vec![0.0; filter_length],
        }
    }

    fn process(&mut self, reference: &[f32], microphone: &[f32]) -> Vec<f32> {
        let mut output = Vec::new();
        for i in 0..microphone.len() {
            // Predict the echo using the adaptive filter
            let predicted_echo = self.predict(&reference[i..i + self.filter_length]);
            // Subtract the predicted echo from the microphone signal
            let cleaned_sample = microphone[i] - predicted_echo;
            output.push(cleaned_sample);
            // Update the filter weights
            self.update(&reference[i..i + self.filter_length], microphone[i]);
        }
        output
    }

    fn predict(&self, reference: &[f32]) -> f32 {
        reference
            .iter()
            .zip(self.weights.iter())
            .map(|(x, w)| x * w)
            .sum()
    }

    fn update(&mut self, reference: &[f32], error: f32) {
        let norm = reference.iter().map(|x| x * x).sum::<f32>();
        if norm > 0.0 {
            for (w, x) in self.weights.iter_mut().zip(reference.iter()) {
                *w += self.step_size * error * x / norm;
            }
        }
    }
}
