// migration/src/main.rs
//
// CLI entry point for sea-orm-migration.
// Run: `cargo run -p migration -- up`

use sea_orm_migration::prelude::*;

#[tokio::main]
async fn main() {
    cli::run_cli(migration::Migrator).await;
}
