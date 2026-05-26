use std::process::Command;
use std::env;
use std::time::Duration;
use std::thread;

fn main() {
    let formats = vec!["gf12", "f32", "bf16", "gf16", "gf8", "posit8", "mxfp8", "nf4", "int8"];
    let seeds = vec!["71", "72", "73"];
    let steps = env::var("TRIOS_STEPS").unwrap_or_else(|_| "50000".into());
    let hidden = env::var("TRIOS_HIDDEN").unwrap_or_else(|_| "256".into());
    let lr = env::var("TRIOS_LR").unwrap_or_else(|_| "0.001".into());
    let optimizer = env::var("TRIOS_OPTIMIZER").unwrap_or_else(|_| "adamw".into());
    let train_data = env::var("TRIOS_TRAIN_DATA").unwrap_or_else(|_| "/work/data/tiny_shakespeare.txt".into());
    let val_data = env::var("TRIOS_VAL_DATA").unwrap_or_else(|_| "/work/data/tiny_shakespeare_val.txt".into());

    for format in &formats {
        for seed in &seeds {
            println!("[sweep] format={format} seed={seed} steps={steps} hidden={hidden} lr={lr}");
            let mut cmd = Command::new("/usr/local/bin/trios-train");
            cmd.arg(format!("--seed={seed}"))
                .arg(format!("--steps={steps}"))
                .arg(format!("--lr={lr}"))
                .arg(format!("--hidden={hidden}"))
                .arg(format!("--optimizer={optimizer}"))
                .arg(format!("--train-data={train_data}"))
                .arg(format!("--val-data={val_data}"))
                .arg(format!("--format={format}"));
            let status = cmd.status().expect("failed to spawn trios-train");
            println!("[sweep] finished format={format} seed={seed} status={status}");
            thread::sleep(Duration::from_secs(5));
        }
    }
    println!("[sweep] all sweeps complete");
}
