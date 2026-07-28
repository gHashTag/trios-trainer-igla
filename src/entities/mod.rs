// src/entities/mod.rs
//
// Re-export all SeaORM entity modules.
//
// Use i64 for every BIGINT column (seed, step, steps_done, id BIGSERIAL, etc.)
// and chrono::DateTime<FixedOffset> (DateTimeWithTimeZone) for TIMESTAMPTZ.

pub mod bpb_samples;
pub mod igla_agents_heartbeat;
pub mod igla_race_trials;
pub mod scarabs;
pub mod strategy_queue;

pub use bpb_samples::Entity as BpbSamples;
pub use igla_agents_heartbeat::Entity as IglaAgentsHeartbeat;
pub use igla_race_trials::Entity as IglaRaceTrials;
pub use scarabs::Entity as Scarabs;
pub use strategy_queue::Entity as StrategyQueue;
