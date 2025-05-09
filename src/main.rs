use anyhow::{Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use dasp_signal::{self as signal, Signal};
use hound;
use ndarray::{Array1, s};
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

mod config {
    // Audio configuration
    pub const SAMPLE_RATE: u32 = 44100;
    pub const CHANNELS: u16 = 1;
    pub const BITS_PER_SAMPLE: u16 = 16;
    pub const FRAME_SIZE: usize = 2048; // Processing frame size
    pub const FILTER_LENGTH: usize = 4096; // Longer filter for better echo modeling
    pub const OVERLAP: usize = FRAME_SIZE / 2; // 50% overlap for processing frames

    // Adaptive filter parameters
    pub const LEARNING_RATE_INITIAL: f32 = 0.05;
    pub const LEARNING_RATE_MIN: f32 = 0.001;
    pub const SPECTRAL_FLOOR: f32 = 1e-6; // Prevents division by zero

    // Echo reduction parameters
    pub const ECHO_SUPPRESSION_DB: f32 = 15.0; // Additional suppression in dB
    pub const ECHO_SUPPRESSION_FACTOR: f32 = 0.177827941; // 10^(-15/20)
}

use config::*;

/// Command line arguments for the Acoustic Echo Cancellation program
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Path to reference audio file (will be created if it doesn't exist)
    #[clap(short, long, default_value = "reference.wav")]
    reference_file: String,
    
    /// Path to output processed audio file
    #[clap(short, long, default_value = "processed.wav")]
    output_file: String,
    
    /// Recording duration in seconds
    #[clap(short, long, default_value_t = 10)]
    duration: u64,
    
    /// Skip user prompts (non-interactive mode)
    #[clap(short, long)]
    non_interactive: bool,
}

/// Processes audio using adaptive filtering to cancel echoes
struct AudioProcessor {
    // Adaptive filter weights in frequency domain
    weights: Vec<Complex<f32>>,
    
    // Reference signal history for frequency domain filtering
    reference_buffer: Array1<f32>,
    
    // Power estimates for normalization
    power_estimate: Vec<f32>,
    
    // FFT related objects
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn RealToComplex<f32>>,
    
    // Buffers for FFT
    fft_input: Vec<f32>,
    fft_output: Vec<Complex<f32>>,
    
    // Double-talk detection
    coherence: f32,
    is_double_talk: bool,
    
    // Noise estimate
    noise_floor: f32,
    
    // Learning rate
    learning_rate: f32,
}

impl AudioProcessor {
    fn new() -> Self {
        let mut planner = RealFftPlanner::new();
        let fft_forward = planner.plan_fft_forward(FRAME_SIZE);
        let fft_inverse = planner.plan_fft_inverse(FRAME_SIZE);

        // Frequency domain has N/2+1 complex values
        let freq_bins = FRAME_SIZE / 2 + 1;

        Self {
            weights: vec![Complex::new(0.0, 0.0); freq_bins],
            reference_buffer: Array1::zeros(FILTER_LENGTH),
            power_estimate: vec![SPECTRAL_FLOOR; freq_bins],
            fft_forward,
            fft_inverse,
            fft_input: vec![0.0; FRAME_SIZE],
            fft_output: vec![Complex::new(0.0, 0.0); freq_bins],
            coherence: 0.0,
            is_double_talk: false,
            noise_floor: 0.001,
            learning_rate: LEARNING_RATE_INITIAL,
        }
    }

    fn process_frame(&mut self, microphone_frame: &mut [f32], reference_frame: &[f32]) -> Result<()> {
        // Update reference buffer (far-end signal)
        self.update_reference_buffer(reference_frame);

        // Perform double-talk detection
        self.detect_double_talk(microphone_frame, reference_frame);

        // Create a windowed frame for processing
        let mut windowed_mic = self.apply_window(microphone_frame);

        // Perform echo cancellation
        if !self.is_double_talk {
            self.cancel_echo(&mut windowed_mic)?;
        }

        // Apply post-processing (noise reduction, etc.)
        self.post_process(&mut windowed_mic);

        // Copy result back to the microphone frame
        microphone_frame.copy_from_slice(&windowed_mic);

        Ok(())
    }

    fn update_reference_buffer(&mut self, reference_frame: &[f32]) {
        // Shift the buffer to make room for new samples
        let buffer_len = self.reference_buffer.len();
        let frame_len = reference_frame.len();

        if frame_len >= buffer_len {
            self.reference_buffer.assign(&Array1::from_vec(
                reference_frame[reference_frame.len() - buffer_len..].to_vec()
            ));
        } else {
            self.reference_buffer.slice_mut(s![0..buffer_len-frame_len])
                .assign(&self.reference_buffer.slice(s![frame_len..]));

            self.reference_buffer.slice_mut(s![buffer_len-frame_len..])
                .assign(&Array1::from_vec(reference_frame.to_vec()));
        }
    }

    fn apply_window(&self, frame: &[f32]) -> Vec<f32> {
        // Apply Hann window to reduce spectral leakage
        let mut result = vec![0.0; frame.len()];
        
        // Use faster windowing calculation
        for (i, sample) in frame.iter().enumerate() {
            let window_val = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / frame.len() as f32).cos());
            result[i] = *sample * window_val;
        }

        result
    }

    fn detect_double_talk(&mut self, mic_frame: &[f32], ref_frame: &[f32]) {
        // Calculate energy levels
        let mic_energy: f32 = mic_frame.iter().map(|&x| x * x).sum::<f32>() / mic_frame.len() as f32;
        let ref_energy: f32 = ref_frame.iter().map(|&x| x * x).sum::<f32>() / ref_frame.len() as f32;

        // Skip detection if reference is too quiet
        if ref_energy < self.noise_floor {
            self.is_double_talk = false;
            return;
        }

        // Calculate normalized cross-correlation
        let mut cross_corr = 0.0;
        let len = std::cmp::min(mic_frame.len(), ref_frame.len());

        for i in 0..len {
            cross_corr += mic_frame[i] * ref_frame[i];
        }
        cross_corr /= (mic_energy * ref_energy).sqrt() * len as f32;

        // Update coherence with smoothing
        self.coherence = 0.9 * self.coherence + 0.1 * cross_corr.abs();

        // Update noise floor estimate (slow adaptation during silence)
        if mic_energy < self.noise_floor * 3.0 {
            self.noise_floor = 0.9 * self.noise_floor + 0.1 * mic_energy;
        }

        // Determine if double-talk is occurring
        // Low coherence suggests near-end speech is occurring simultaneously
        self.is_double_talk = self.coherence < 0.5 && mic_energy > ref_energy * 0.1;

        // Adjust learning rate based on detection
        if self.is_double_talk {
            self.learning_rate = LEARNING_RATE_MIN; // Slow adaptation during double-talk
        } else {
            // Gradually increase learning rate when no double-talk
            self.learning_rate = (self.learning_rate * 0.95 + LEARNING_RATE_INITIAL * 0.05)
                .clamp(LEARNING_RATE_MIN, LEARNING_RATE_INITIAL);
        }
    }

    fn cancel_echo(&mut self, frame: &mut [f32]) -> Result<()> {
        // Copy frame to FFT input buffer
        self.fft_input[..frame.len()].copy_from_slice(frame);

        // Perform FFT on microphone signal
        let mut mic_spectrum = vec![Complex::new(0.0, 0.0); self.fft_output.len()];
        self.fft_forward.process(&mut self.fft_input, &mut mic_spectrum)
            .context("Failed to perform FFT on microphone signal")?;

        // For each frame in the reference buffer that overlaps with our current frame
        let mut echo_spectrum = vec![Complex::new(0.0, 0.0); mic_spectrum.len()];

        // Get the most recent part of reference buffer for processing
        let recent_reference = self.reference_buffer.slice(s![self.reference_buffer.len() - FRAME_SIZE..]).to_vec();

        // Perform FFT on reference signal
        let mut ref_input = vec![0.0; FRAME_SIZE];
        ref_input[..recent_reference.len()].copy_from_slice(&recent_reference);

        self.fft_forward.process(&mut ref_input, &mut self.fft_output)
            .context("Failed to perform FFT on reference signal")?;

        // Calculate estimated echo using filter weights and update power estimates in single pass
        for i in 0..echo_spectrum.len() {
            echo_spectrum[i] = self.fft_output[i] * self.weights[i];
            
            let ref_power = self.fft_output[i].norm_sqr();
            self.power_estimate[i] = 0.9 * self.power_estimate[i] + 0.1 * ref_power;
            self.power_estimate[i] = self.power_estimate[i].max(SPECTRAL_FLOOR);
        }

        // Calculate error and update filter weights in frequency domain (NLMS algorithm)
        if !self.is_double_talk {
            for i in 0..self.weights.len() {
                let error = mic_spectrum[i] - echo_spectrum[i];
                
                // Normalized update with spectral floor
                let normalized_lr = self.learning_rate / self.power_estimate[i];
                
                // Weight update
                self.weights[i] = self.weights[i] + normalized_lr * error * self.fft_output[i].conj();
            }
        }

        // Subtract echo from microphone signal
        let mut error_spectrum = vec![Complex::new(0.0, 0.0); mic_spectrum.len()];
        for i in 0..error_spectrum.len() {
            error_spectrum[i] = mic_spectrum[i] - echo_spectrum[i];
        }

        // Transform back to time domain
        let mut output_buffer = vec![0.0; FRAME_SIZE];
        self.fft_inverse.process(&mut error_spectrum, &mut output_buffer)
            .context("Failed to perform inverse FFT")?;

        // Scale the output (IFFT normalization) and copy to frame
        let scale_factor = 1.0 / FRAME_SIZE as f32;
        for i in 0..frame.len() {
            frame[i] = output_buffer[i] * scale_factor;
        }

        Ok(())
    }

    fn post_process(&self, frame: &mut [f32]) {
        // Apply additional echo suppression
        if !self.is_double_talk {
            for sample in frame.iter_mut() {
                *sample *= ECHO_SUPPRESSION_FACTOR;
            }
        }

        // Simple noise gate
        for sample in frame.iter_mut() {
            if sample.abs() < self.noise_floor * 2.0 {
                *sample *= 0.5; // Attenuate low-level signals that might be residual echo
            }
        }
    }
}

/// Handles recording audio to a WAV file
struct AudioRecorder {
    spec: hound::WavSpec,
    writer: Arc<Mutex<hound::WavWriter<BufWriter<File>>>>,
}

impl AudioRecorder {
    fn new(file_name: &str, sample_rate: u32, channels: u16) -> Result<Self> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: BITS_PER_SAMPLE,
            sample_format: hound::SampleFormat::Int,
        };

        let writer = hound::WavWriter::create(file_name, spec)
            .with_context(|| format!("Failed to create WAV file: {}", file_name))?;

        Ok(Self {
            spec,
            writer: Arc::new(Mutex::new(writer)),
        })
    }

    fn write_samples(&self, samples: &[f32]) -> Result<()> {
        let mut writer = self.writer.lock().unwrap();
        for &sample in samples {
            let amplitude = (sample * i16::MAX as f32) as i16;
            writer.write_sample(amplitude)
                .context("Failed to write sample to WAV file")?;
        }
        Ok(())
    }

    fn finalize(self) -> Result<()> {
        let writer = Arc::try_unwrap(self.writer)
            .map_err(|_| anyhow::anyhow!("Could not unwrap writer Arc"))?
            .into_inner()
            .map_err(|_| anyhow::anyhow!("Could not unlock writer mutex"))?;

        writer.finalize().context("Failed to finalize WAV file")?;
        Ok(())
    }
}

/// Manages audio playback
struct AudioPlayer {
    stream: cpal::Stream,
    done_playing: Arc<Mutex<bool>>,
}

impl AudioPlayer {
    fn new(device: &cpal::Device, config: &cpal::SupportedStreamConfig, samples: Vec<f32>, channels: usize) -> Result<Self> {
        let done_playing = Arc::new(Mutex::new(false));
        let done_clone = done_playing.clone();

        let samples = Arc::new(Mutex::new(samples.into_iter()));

        let stream = device.build_output_stream(
            &config.config(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut samples_lock = samples.lock().unwrap();

                let mut all_done = true;
                for frame in data.chunks_mut(channels) {
                    if let Some(sample) = samples_lock.next() {
                        all_done = false;
                        // Fill all channels with the same sample
                        for out in frame.iter_mut() {
                            *out = sample;
                        }
                    } else {
                        // End of file, fill with silence
                        for out in frame.iter_mut() {
                            *out = 0.0;
                        }
                    }
                }

                if all_done {
                    *done_clone.lock().unwrap() = true;
                }
            },
            |err| {
                eprintln!("Playback error: {}", err);
            },
            None,
        )?;

        Ok(Self {
            stream,
            done_playing,
        })
    }

    fn play(&self) -> Result<()> {
        self.stream.play().context("Failed to start playback")?;
        Ok(())
    }

    fn is_done(&self) -> bool {
        *self.done_playing.lock().unwrap()
    }
}

/// Record audio from microphone while applying echo cancellation
fn record_with_aec(
    mic_device: &cpal::Device,
    mic_config: &cpal::SupportedStreamConfig,
    reference_samples: &[f32],
    output_file: &str,
    duration_secs: u64
) -> Result<()> {
    // Create recorder for processed audio
    let recorder = AudioRecorder::new(
        output_file,
        mic_config.sample_rate().0,
        mic_config.channels() as u16
    )?;
    let recorder = Arc::new(recorder);

    // Create audio processor
    let processor = Arc::new(Mutex::new(AudioProcessor::new()));

    // Create a ring buffer for reference samples
    let mut ref_buffer = signal::from_iter(reference_samples.iter().cloned().cycle());

    // Set up processing buffers
    let frame_len = FRAME_SIZE - OVERLAP;
    let mut mic_buffer = vec![0.0; frame_len];
    let mut ref_frame = vec![0.0; frame_len];

    // Set up input stream for microphone
    let processor_clone = processor.clone();
    let recorder_clone = recorder.clone();

    let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        // Copy input data to our buffer
        let len = std::cmp::min(data.len(), mic_buffer.len());
        mic_buffer[..len].copy_from_slice(&data[..len]);

        // Get next chunk of reference audio
        for i in 0..frame_len {
            ref_frame[i] = ref_buffer.next();
        }

        // Process the audio
        let mut processor_lock = processor_clone.lock().unwrap();
        if let Err(e) = processor_lock.process_frame(&mut mic_buffer, &ref_frame) {
            eprintln!("Processing error: {}", e);
        }

        // Record the processed audio
        if let Err(e) = recorder_clone.write_samples(&mic_buffer) {
            eprintln!("Recording error: {}", e);
        }
    };

    let err_fn = |err| {
        eprintln!("Stream error: {}", err);
    };

    // Build and play input stream
    let input_stream = mic_device.build_input_stream(
        &mic_config.config(),
        input_data_fn,
        err_fn,
        None
    )?;

    input_stream.play()?;

    // Wait for the specified duration
    println!("Recording and processing for {} seconds...", duration_secs);
    std::thread::sleep(Duration::from_secs(duration_secs));

    // Stop recording
    drop(input_stream);

    // Finalize the WAV file
    Arc::try_unwrap(recorder)
        .map_err(|_| anyhow::anyhow!("Could not unwrap recorder Arc"))?
        .finalize()?;

    println!("Processing complete! Saved as {}", output_file);
    Ok(())
}

/// Load a WAV file into memory
fn load_wav_file(file_path: &str) -> Result<Vec<f32>> {
    println!("Loading reference audio: {}", file_path);

    let reader = hound::WavReader::open(file_path)
        .with_context(|| format!("Failed to open WAV file: {}", file_path))?;

    let samples: Vec<f32> = match reader.spec().sample_format {
        hound::SampleFormat::Int => {
            reader.into_samples::<i16>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / i16::MAX as f32)
                .collect()
        },
        hound::SampleFormat::Float => {
            reader.into_samples::<f32>()
                .filter_map(Result::ok)
                .collect()
        }
    };

    println!("Loaded {} samples", samples.len());
    Ok(samples)
}

/// Play audio from a WAV file
fn play_audio(file_name: &str, wait_for_completion: bool) -> Result<()> {
    println!("Playing audio: {}", file_name);

    let host = cpal::default_host();
    let device = host.default_output_device()
        .context("No output device found")?;

    let config = device.default_output_config()?;
    let channels = config.channels() as usize;

    // Load the audio file
    let samples = load_wav_file(file_name)?;

    // Create the player
    let player = AudioPlayer::new(&device, &config, samples, channels)?;
    player.play()?;

    if wait_for_completion {
        // Wait for playback to complete
        println!("Waiting for playback to complete...");
        while !player.is_done() {
            std::thread::sleep(Duration::from_millis(100));
        }
    } else {
        // Just wait a few seconds
        let duration = Duration::from_secs(5);
        std::thread::sleep(duration);
    }

    Ok(())
}

/// Run the complete echo cancellation system
fn run_echo_cancellation_system(reference_file: &str, output_file: &str, duration_secs: u64) -> Result<()> {
    // Load reference audio
    let reference_samples = load_wav_file(reference_file)?;

    // Set up audio devices
    let host = cpal::default_host();

    let input_device = host.default_input_device()
        .context("No input device found")?;
    let output_device = host.default_output_device()
        .context("No output device found")?;

    let input_config = input_device.default_input_config()?;
    let output_config = output_device.default_output_config()?;

    println!("Input: {} channels @ {} Hz", input_config.channels(), input_config.sample_rate().0);
    println!("Output: {} channels @ {} Hz", output_config.channels(), output_config.sample_rate().0);

    // Set up playback for reference audio
    let player = AudioPlayer::new(
        &output_device,
        &output_config,
        reference_samples.clone(),
        output_config.channels() as usize
    )?;

    // Start playback
    player.play()?;

    // Record with echo cancellation
    record_with_aec(
        &input_device,
        &input_config,
        &reference_samples,
        output_file,
        duration_secs
    )?;

    Ok(())
}

/// Creates a test tone file if no reference audio is available
fn create_test_tone_file(file_path: &str) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: BITS_PER_SAMPLE,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(file_path, spec)?;

    // Create a 5-second test tone with varying frequencies
    let duration_samples = 5 * SAMPLE_RATE;
    let amplitude = 0.7 * i16::MAX as f32;

    for t in 0..duration_samples {
        let time = t as f32 / SAMPLE_RATE as f32;

        // Create a mix of frequencies that sweep up and down
        let freq1 = 300.0 + 700.0 * (time / 5.0).sin();
        let freq2 = 800.0 + 400.0 * ((time + 2.0) / 3.0).sin();

        let sample = 0.5 * (2.0 * std::f32::consts::PI * freq1 * time).sin() +
            0.3 * (2.0 * std::f32::consts::PI * freq2 * time).sin();

        writer.write_sample((sample * amplitude) as i16)?;
    }

    writer.finalize()?;
    Ok(())
}

/// Wait for user input in interactive mode
fn wait_for_user_input(non_interactive: bool, message: &str) -> Result<()> {
    if non_interactive {
        return Ok(());
    }

    println!("\n{}", message);
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(())
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();
    
    println!("===== Advanced Acoustic Echo Cancellation System =====");
    println!("This program will record audio from your microphone while");
    println!("playing back audio from '{}', and apply echo cancellation.", args.reference_file);
    println!("Make sure your speakers and microphone are set up correctly.");

    // Check if reference file exists
    if !Path::new(&args.reference_file).exists() {
        println!("\nReference file not found. Creating a test tone file...");
        create_test_tone_file(&args.reference_file)?;
        println!("Created test tone file: {}", args.reference_file);
    }

    // Start the echo cancellation process
    wait_for_user_input(args.non_interactive, "Press Enter to start echo cancellation process...")?;

    // Run the echo cancellation system
    run_echo_cancellation_system(&args.reference_file, &args.output_file, args.duration)?;

    // Play the processed audio
    wait_for_user_input(args.non_interactive, "Echo cancellation complete. Press Enter to play the processed audio...")?;
    play_audio(&args.output_file, true)?;

    println!("\nEcho cancellation process finished!");
    Ok(())
}
