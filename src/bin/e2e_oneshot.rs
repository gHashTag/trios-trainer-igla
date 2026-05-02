//! E2E ONESHOT — Local scarab test (30 min → green or diagnosis)
//!
//! Run: cargo run --bin e2e_oneshot
//!
//! This binary performs end-to-end testing of the scarab pipeline locally,
//! without Railway, without MCP, in a single container.

use std::env;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

#[derive(Debug)]
struct Probe {
    id: i64,
    canon_name: String,
}

#[derive(Debug)]
struct ProbeStatus {
    id: i64,
    canon_name: String,
    status: String,
    claimed_at: Option<String>,
    started_at: Option<String>,
    finished_at: Option<String>,
    final_bpb: Option<f64>,
    final_step: Option<i32>,
    last_error: Option<String>,
    attempts: i32,
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  E2E ONESHOT — Local scarab test                                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP 0: Load env
    // ═══════════════════════════════════════════════════════════════════════════════
    println!("\n📋 Loading environment variables...");

    let neon_dsn = env::var("NEON_DATABASE_URL")
        .or_else(|_| env::var("DATABASE_URL"))
        .expect("NEON_DATABASE_URL or DATABASE_URL must be set");

    println!("   NEON_DATABASE_URL: {}...", &neon_dsn[..neon_dsn.len().min(30)]);

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP 1: Verify Neon connection
    // ═══════════════════════════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  STEP 1: Verify Neon connection                                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    let output = Command::new("psql")
        .arg(&neon_dsn)
        .arg("-c")
        .arg("SELECT now();")
        .output();

    match output {
        Ok(out) if out.status.success() => {
            println!("✅ Neon connection OK");
            let time = String::from_utf8_lossy(&out.stdout);
            println!("   {}", time.lines().next().unwrap_or("now()"));
        }
        Ok(out) => {
            eprintln!("❌ ERROR: psql returned non-zero");
            eprintln!("   stdout: {}", String::from_utf8_lossy(&out.stdout));
            eprintln!("   stderr: {}", String::from_utf8_lossy(&out.stderr));
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("❌ ERROR: Failed to execute psql: {}", e);
            eprintln!("   Is psql installed and in PATH?");
            std::process::exit(1);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP 2: Pull image and verify contents
    // ═══════════════════════════════════════════════════════════════════════════════
    let image = "ghcr.io/ghashtag/trios-trainer-igla:956c6c9542b0924c20a01f3fc70580dee9ef58d5";

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  STEP 2: Pull image and verify contents                                     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    println!("📥 Pulling image: {}...", &image[..image.len().min(50)]);

    let pull_output = Command::new("docker")
        .arg("pull")
        .arg(image)
        .output();

    match pull_output {
        Ok(out) if out.status.success() => {
            println!("✅ Image pulled successfully");
        }
        Ok(out) => {
            eprintln!("❌ ERROR: docker pull failed");
            eprintln!("   stderr: {}", String::from_utf8_lossy(&out.stderr));
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("❌ ERROR: Failed to execute docker: {}", e);
            eprintln!("   Is docker installed and running?");
            std::process::exit(1);
        }
    }

    // Verify image contents
    println!("\n🔍 Verifying image contents...");

    let check_bins = Command::new("docker")
        .arg("run")
        .arg("--rm")
        .arg("--entrypoint")
        .arg("sh")
        .arg(image)
        .arg("-c")
        .arg("ls /usr/local/bin/ | grep -E 'scarab|trios' || echo 'NOT FOUND'")
        .output();

    match check_bins {
        Ok(out) => {
            let bins = String::from_utf8_lossy(&out.stdout);
            println!("   Binaries found:");
            for line in bins.lines() {
                if !line.is_empty() && line != "NOT FOUND" {
                    println!("     - {}", line);
                }
            }

            if !bins.contains("scarab") {
                eprintln!("❌ ERROR: scarab binary NOT FOUND in image");
                eprintln!("   The image may be from an older commit.");
                std::process::exit(1);
            }

            if !bins.contains("trios-train") {
                println!("⚠️  WARNING: trios-train binary NOT FOUND in image");
            }
        }
        Err(e) => {
            eprintln!("❌ ERROR: Failed to verify image: {}", e);
            std::process::exit(1);
        }
    }

    println!("✅ Image verification complete");

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP 3: Stop any existing scarab-local container and start new one
    // ═══════════════════════════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  STEP 3: Start scarab container                                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    // Stop existing container if running
    let _ = Command::new("docker")
        .args(["stop", "scarab-local"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    let _ = Command::new("docker")
        .args(["rm", "scarab-local"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    let scarab_id = format!("local-mac-{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs());

    println!("🚀 Starting scarab container...");
    println!("   SCARAB_ID: {}", scarab_id);
    println!("   RUST_LOG: debug");

    let run_output = Command::new("docker")
        .args([
            "run", "-d", "--name", "scarab-local",
            "-e", &format!("NEON_DATABASE_URL={}", neon_dsn),
            "-e", &format!("TRIOS_NEON_DSN={}", neon_dsn),
            "-e", &format!("DATABASE_URL={}", neon_dsn),
            "-e", "ACCOUNT=local-dev",
            "-e", &format!("SCARAB_ID={}", scarab_id),
            "-e", "RUST_LOG=debug",
            image,
            "scarab",
        ])
        .output();

    match run_output {
        Ok(out) if out.status.success() => {
            println!("✅ Scarab container started (detached)");
        }
        Ok(out) => {
            eprintln!("❌ ERROR: Failed to start scarab container");
            eprintln!("   stderr: {}", String::from_utf8_lossy(&out.stderr));
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("❌ ERROR: Failed to execute docker: {}", e);
            std::process::exit(1);
        }
    }

    // Wait a moment for startup
    thread::sleep(Duration::from_secs(3));

    // Check if container is still running
    let ps_output = Command::new("docker")
        .args(["ps", "--format", "{{.Names}}"])
        .output();

    match ps_output {
        Ok(out) => {
            let running = String::from_utf8_lossy(&out.stdout);
            if !running.contains("scarab-local") {
                eprintln!("❌ ERROR: Scarab container crashed immediately!");
                eprintln!("📋 Crash log:");

                let logs = Command::new("docker")
                    .args(["logs", "scarab-local"])
                    .output();
                if let Ok(log_out) = logs {
                    eprintln!("{}", String::from_utf8_lossy(&log_out.stdout));
                    eprintln!("{}", String::from_utf8_lossy(&log_out.stderr));
                }
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("❌ ERROR: Failed to check container status: {}", e);
            std::process::exit(1);
        }
    }

    // Show initial logs
    println!("\n📋 Initial scarab logs (last 10 lines):");
    let logs = Command::new("docker")
        .args(["logs", "--tail", "10", "scarab-local"])
        .output();

    if let Ok(out) = logs {
        println!("{}", String::from_utf8_lossy(&out.stdout));
    }

    println!("✅ Scarab container running");

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP 4: Insert probe into strategy_queue
    // ═══════════════════════════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  STEP 4: Insert probe into strategy_queue                                    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    let insert_sql = r#"
INSERT INTO strategy_queue (
  canon_name, status, config_json, steps_budget, kill_at_step, kill_if_bpb_over, timeout_seconds
) VALUES (
  'PROBE-ONESHOT-LOCAL-' || extract(epoch from now())::int,
  'pending',
  '{"data": {"corpus": "tiny_shakespeare", "train_path": "/work/data/tiny_shakespeare.txt", "val_path": "/work/data/tiny_shakespeare.txt"},
    "lr": 0.004, "hidden": 384, "steps": 500, "seed": 1597, "format": "fp32"}'::jsonb,
  500, 250, 12.0, 600
)
RETURNING id, canon_name;
"#;

    let insert_output = Command::new("psql")
        .arg(&neon_dsn)
        .arg("-c")
        .arg(insert_sql)
        .output();

    let probe = match insert_output {
        Ok(out) if out.status.success() => {
            let result = String::from_utf8_lossy(&out.stdout);
            // Parse: "   123 | PROBE-ONESHOT-LOCAL-..."
            let lines: Vec<&str> = result.lines().collect();
            if lines.len() >= 3 {
                let parts: Vec<&str> = lines[2].split('|').collect();
                let id: i64 = parts[0].trim().parse().unwrap_or(0);
                let canon_name = parts[1].trim().to_string();
                Probe { id, canon_name }
            } else {
                eprintln!("❌ ERROR: Unexpected psql output format");
                eprintln!("   {}", result);
                std::process::exit(1);
            }
        }
        Ok(out) => {
            eprintln!("❌ ERROR: Failed to insert probe");
            eprintln!("   stderr: {}", String::from_utf8_lossy(&out.stderr));
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("❌ ERROR: Failed to execute psql: {}", e);
            std::process::exit(1);
        }
    };

    println!("✅ Probe inserted");
    println!("   ID: {}", probe.id);
    println!("   Canon name: {}", probe.canon_name);
    println!("   Config: hidden=384, steps=500, lr=0.004, seed=1597");
    println!("   Expected runtime: ~1-2 minutes on CPU");

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP 5: Watch scarab claim and run (auto-polling 60s)
    // ═══════════════════════════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  STEP 5: Watch scarab claim and run (auto-polling 60s)                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!("");
    println!("Expected timeline:");
    println!("  T+0:10  → status: pending → running, claimed_at filled");
    println!("  T+0:30  → bpb_samples row at step=100 (if eval_every=100)");
    println!("  T+1:00  → bpb_samples row at step=250 (kill_at_step)");
    println!("  T+1:30  → bpb_samples row at step=500");
    println!("  T+1:35  → status: running → done, final_bpb non-NULL");
    println!("");
    println!("Polling every 5 seconds for 12 cycles (60 seconds)...\n");

    for i in 1..=12 {
        println!("═══ Poll cycle {}/12 ═══", i);

        // Get status
        let status_sql = format!(
            r#"
SELECT id, canon_name, status,
       COALESCE(claimed_at::text, 'NULL') as claimed_at,
       COALESCE(started_at::text, 'NULL') as started_at,
       COALESCE(finished_at::text, 'NULL') as finished_at,
       COALESCE(final_bpb::text, 'NULL') as final_bpb,
       COALESCE(final_step::text, 'NULL') as final_step,
       COALESCE(last_error, 'NULL') as last_error,
       attempts
FROM strategy_queue WHERE id = {};
"#,
            probe.id
        );

        let status_output = Command::new("psql")
            .arg(&neon_dsn)
            .arg("-c")
            .arg(&status_sql)
            .output();

        let status = match status_output {
            Ok(out) => {
                let result = String::from_utf8_lossy(&out.stdout);
                parse_probe_status(&result, probe.id)
            }
            Err(_) => None,
        };

        if let Some(s) = status {
            println!("📊 Status:");
            println!("   status: {}", s.status);
            println!("   claimed_at: {}", s.claimed_at.as_deref().unwrap_or("NULL"));
            println!("   started_at: {}", s.started_at.as_deref().unwrap_or("NULL"));
            println!("   final_step: {}", s.final_step.map(|x| x.to_string()).unwrap_or("NULL".to_string()));
            println!("   final_bpb: {}", s.final_bpb.map(|x| x.to_string()).unwrap_or("NULL".to_string()));

            if s.status == "done" {
                println!("\n🎉 Probe completed!");
                break;
            } else if s.status == "failed" {
                println!("\n❌ Probe failed!");
                if let Some(err) = &s.last_error {
                    println!("   Error: {}", err);
                }
                break;
            }
        }

        // Get bpb_samples count
        let count_sql = format!(
            "SELECT COUNT(*) FROM bpb_samples WHERE canon_name = '{}'",
            probe.canon_name
        );

        let count_output = Command::new("psql")
            .arg(&neon_dsn)
            .arg("-t")
            .arg("-c")
            .arg(&count_sql)
            .output();

        if let Ok(out) = count_output {
            let count = String::from_utf8_lossy(&out.stdout).trim().to_string();
            println!("   bpb_samples count: {}", count);
        }

        // Show latest logs
        let logs = Command::new("docker")
            .args(["logs", "--tail", "5", "scarab-local"])
            .output();

        if let Ok(out) = logs {
            let log_output = String::from_utf8_lossy(&out.stdout);
            if !log_output.trim().is_empty() {
                println!("\n📋 Latest scarab logs:");
                for line in log_output.lines().take(3) {
                    println!("   {}", line);
                }
            }
        }

        println!();

        if i < 12 && status.as_ref().map(|s| s.status == "running" || s.status == "pending").unwrap_or(true) {
            thread::sleep(Duration::from_secs(5));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // STEP 6: Verdict
    // ═══════════════════════════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  STEP 6: Verdict                                                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    let verdict_sql = format!(
        r#"
SELECT
  CASE
    WHEN final_bpb IS NOT NULL AND final_step = 500 THEN '✅ E2E PASS — trainer + scarab + Neon writer ALL work'
    WHEN final_bpb IS NULL AND status = 'done' THEN '❌ Bug E persists — final_bpb NULL despite done status'
    WHEN status = 'failed' THEN '❌ Trainer crashed — last_error has details'
    WHEN status = 'pending' THEN '❌ Scarab never claimed — check scarab logs'
    WHEN status = 'running' THEN '⏳ Still running — may be slow on CPU or stuck'
    ELSE '❌ Unknown state'
  END AS verdict,
  status,
  claimed_at IS NOT NULL as claimed,
  started_at IS NOT NULL as started,
  finished_at IS NOT NULL as finished,
  final_bpb,
  final_step,
  (SELECT COUNT(*) FROM bpb_samples WHERE canon_name = strategy_queue.canon_name) as bpb_sample_count,
  last_error,
  attempts
FROM strategy_queue WHERE id = {};
"#,
        probe.id
    );

    let verdict_output = Command::new("psql")
        .arg(&neon_dsn)
        .arg("-c")
        .arg(&verdict_sql)
        .output();

    match verdict_output {
        Ok(out) => {
            println!("{}", String::from_utf8_lossy(&out.stdout));
        }
        Err(e) => {
            eprintln!("❌ ERROR: Failed to get verdict: {}", e);
        }
    }

    // Show full probe row
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Full probe row                                                             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    let full_sql = format!(
        r#"
SELECT id, canon_name, status,
       claimed_at, started_at, finished_at,
       final_bpb, final_step, last_error, attempts,
       config_json::text
FROM strategy_queue WHERE id = {};
"#,
        probe.id
    );

    let _ = Command::new("psql")
        .arg(&neon_dsn)
        .arg("-c")
        .arg(&full_sql)
        .status();

    // Show bpb_samples
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  BPB Samples                                                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    let samples_sql = format!(
        "SELECT * FROM bpb_samples WHERE canon_name = '{}' ORDER BY step",
        probe.canon_name
    );

    let _ = Command::new("psql")
        .arg(&neon_dsn)
        .arg("-c")
        .arg(&samples_sql)
        .status();

    // Show full scarab logs
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Full scarab logs (last 50 lines)                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    let logs = Command::new("docker")
        .args(["logs", "--tail", "50", "scarab-local"])
        .output();

    if let Ok(out) = logs {
        println!("{}", String::from_utf8_lossy(&out.stdout));
        eprintln!("{}", String::from_utf8_lossy(&out.stderr));
    }

    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    println!("E2E ONESHOT COMPLETE");
    println!();
    println!("To view logs anytime: docker logs scarab-local");
    println!("To stop container: docker stop scarab-local && docker rm scarab-local");
    println!("═══════════════════════════════════════════════════════════════════════════════");
}

fn parse_probe_status(output: &str, expected_id: i64) -> Option<ProbeStatus> {
    // Parse psql table output
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() < 4 {
        return None;
    }

    // Skip header and separator lines, find data row
    for line in lines.iter().skip(2) {
        if line.contains("---") {
            continue;
        }

        // Parse: "  123 | PROBE-... | pending | ..."
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() >= 3 {
            let id: i64 = parts.get(0)?.trim().parse().ok()?;
            if id != expected_id {
                continue;
            }

            let canon_name = parts.get(1)?.trim().to_string();
            let status = parts.get(2)?.trim().to_string();
            let claimed_at = parts.get(3).and_then(|s| {
                let trimmed = s.trim();
                if trimmed == "NULL" || trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            });
            let started_at = parts.get(4).and_then(|s| {
                let trimmed = s.trim();
                if trimmed == "NULL" || trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            });
            let finished_at = parts.get(5).and_then(|s| {
                let trimmed = s.trim();
                if trimmed == "NULL" || trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            });
            let final_bpb = parts.get(6).and_then(|s| {
                let trimmed = s.trim();
                if trimmed == "NULL" || trimmed.is_empty() {
                    None
                } else {
                    trimmed.parse().ok()
                }
            });
            let final_step = parts.get(7).and_then(|s| {
                let trimmed = s.trim();
                if trimmed == "NULL" || trimmed.is_empty() {
                    None
                } else {
                    trimmed.parse().ok()
                }
            });
            let last_error = parts.get(8).and_then(|s| {
                let trimmed = s.trim();
                if trimmed == "NULL" || trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            });
            let attempts: i32 = parts.get(9)
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);

            return Some(ProbeStatus {
                id,
                canon_name,
                status,
                claimed_at,
                started_at,
                finished_at,
                final_bpb,
                final_step,
                last_error,
                attempts,
            });
        }
    }

    None
}
