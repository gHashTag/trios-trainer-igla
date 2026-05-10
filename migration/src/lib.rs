// migration/src/lib.rs
//
// SeaORM migration crate for trios-trainer-igla.
//
// Anchor: phi^2 + phi^-2 = 3 · DOI 10.5281/zenodo.19227877

pub use sea_orm_migration::prelude::*;

pub mod m20260509_000001_init_schema;
pub mod m20260510_000001_step_to_bigint;
pub mod m20260510_000002_bpb_samples_ema_bpb;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20260509_000001_init_schema::Migration),
            Box::new(m20260510_000001_step_to_bigint::Migration),
            Box::new(m20260510_000002_bpb_samples_ema_bpb::Migration),
        ]
    }
}
