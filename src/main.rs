use cpal::traits::{HostTrait, DeviceTrait, StreamTrait};

fn main() {
    let host = cpal::default_host();

    let device = host.default_input_device().expect("No input device found.");
    print!("- {}", device.name().unwrap());

    let config = device.default_input_config().unwrap();
    println!("- {:?}", config);

    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            println!("{:?}", &data[0..10]);
        }, move |error| {
            eprintln!("Error occurred: {}", error);
        },
        None
    ).unwrap();

    stream.play().unwrap();

    println!("Listening... Press Ctrl+C to stop.");
    std::thread::sleep(std::time::Duration::from_secs(2));
}
