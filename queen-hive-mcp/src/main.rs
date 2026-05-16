//! Queen-Hive MCP server — JSON-RPC 2.0 over stdio.
//!
//! Reads requests from stdin, writes responses to stdout, logs to stderr.
//! Each request:
//!   {"jsonrpc":"2.0","id":N,"method":"<tool>","params":{...}}
//!
//! Anchor: phi^2 + phi^-2 = 3

use anyhow::{anyhow, Context, Result};
use queen_hive_mcp::{
    bump_strategy, connect_pg, emergency_mass_op, fleet_status, kill_scarab, pause_scarab,
    resume_scarab, spawn_scarab, BumpArgs, CanonOnly, KillArgs, MassOpArgs, SpawnArgs,
};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,queen_hive_mcp=debug".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let pg_url = std::env::var("DATABASE_URL")
        .context("DATABASE_URL required — Queen-Hive only writes to ssot, no Railway API")?;
    let pg = connect_pg(&pg_url).await?;

    tracing::info!("queen-hive-mcp ready · 7 tools · NO RAILWAY API");

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = reader.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let err = json!({
                    "jsonrpc":"2.0","id":null,
                    "error":{"code":-32700,"message":format!("parse error: {e}")}
                });
                stdout.write_all(format!("{err}\n").as_bytes()).await?;
                stdout.flush().await?;
                continue;
            }
        };

        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(Value::Null);

        let result = handle(&pg, method, params).await;
        let resp = match result {
            Ok(data) => json!({"jsonrpc":"2.0","id":id,"result":data}),
            Err(e)   => json!({
                "jsonrpc":"2.0","id":id,
                "error":{"code":-32000,"message":e.to_string()}
            }),
        };
        stdout.write_all(format!("{resp}\n").as_bytes()).await?;
        stdout.flush().await?;
    }
    Ok(())
}

async fn handle(
    pg: &tokio_postgres::Client,
    method: &str,
    params: Value,
) -> Result<Value> {
    match method {
        "spawn_scarab"      => spawn_scarab(pg, serde_json::from_value::<SpawnArgs>(params)?).await,
        "bump_strategy"     => bump_strategy(pg, serde_json::from_value::<BumpArgs>(params)?).await,
        "pause_scarab"      => pause_scarab(pg, serde_json::from_value::<CanonOnly>(params)?).await,
        "resume_scarab"     => resume_scarab(pg, serde_json::from_value::<CanonOnly>(params)?).await,
        "kill_scarab"       => kill_scarab(pg, serde_json::from_value::<KillArgs>(params)?).await,
        "fleet_status"      => fleet_status(pg).await,
        "emergency_mass_op" => emergency_mass_op(pg, serde_json::from_value::<MassOpArgs>(params)?).await,
        "tools/list"        => Ok(json!({
            "tools": [
                "spawn_scarab","bump_strategy","pause_scarab","resume_scarab",
                "kill_scarab","fleet_status","emergency_mass_op"
            ]
        })),
        _ => Err(anyhow!("unknown method '{method}'")),
    }
}
