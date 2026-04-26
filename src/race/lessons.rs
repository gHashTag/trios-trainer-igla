//! Failure Memory — automatic lesson generation from pruned trials

use serde::{Deserialize, Serialize};
use crate::neon::NeonDb;
use anyhow::Result;
use uuid::Uuid;

/// Lesson type for categorizing patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LessonType {
    Avoid,    // "AVOID: ..." — definite anti-pattern
    Pattern,  // "PATTERN: ..." — observed pattern
    Winner,   // "WINNER: ..." — successful config
    Warn,     // "WARN: ..." — potential issue
    Info,     // "INFO: ..." — neutral observation
}

impl std::fmt::Display for LessonType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LessonType::Avoid => write!(f, "AVOID"),
            LessonType::Pattern => write!(f, "PATTERN"),
            LessonType::Winner => write!(f, "WINNER"),
            LessonType::Warn => write!(f, "WARN"),
            LessonType::Info => write!(f, "INFO"),
        }
    }
}

/// Trial outcome
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Outcome {
    Pruned,
    Failed,
    Slow,
    Unstable,
    Timeout,
}

impl std::fmt::Display for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Outcome::Pruned => write!(f, "pruned"),
            Outcome::Failed => write!(f, "failed"),
            Outcome::Slow => write!(f, "slow"),
            Outcome::Unstable => write!(f, "unstable"),
            Outcome::Timeout => write!(f, "timeout"),
        }
    }
}

/// Trial configuration for analysis
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrialConfig {
    pub lr: Option<f64>,
    pub d_model: Option<usize>,
    pub hidden: Option<usize>,
    pub n_layers: Option<usize>,
    pub optimizer: Option<String>,
    pub activation: Option<String>,
    pub weight_decay: Option<f64>,
    pub dropout: Option<f64>,
    pub warmup_steps: Option<usize>,
    pub max_steps: Option<usize>,
}

/// ASHA rung data
#[derive(Debug, Clone)]
pub struct RungData {
    pub step: usize,
    pub bpb: f64,
}

/// Generate lesson from pruned trial
pub fn generate_lesson(
    config: &TrialConfig,
    rung: &RungData,
    outcome: Outcome,
) -> (String, LessonType) {
    let mut lessons = Vec::new();

    // Analyze learning rate
    if let Some(lr) = config.lr {
        if lr > 0.05 {
            lessons.push((
                format!("AVOID: lr={} too high — BPB={} at rung-{}, always pruned",
                         lr, rung.bpb, rung.step),
                LessonType::Avoid
            ));
        } else if lr < 0.001 {
            lessons.push((
                format!("WARN: lr={} too low — may not converge", lr),
                LessonType::Warn
            ));
        }
    }

    // Analyze model capacity
    if let Some(d_model) = config.d_model {
        if d_model <= 64 && rung.bpb > 2.8 {
            lessons.push((
                format!("PATTERN: d_model={} insufficient capacity, never below {} BPB",
                         d_model, rung.bpb),
                LessonType::Pattern
            ));
        }
    }

    // Analyze hidden size
    if let Some(hidden) = config.hidden {
        if hidden < 32 {
            lessons.push((
                format!("WARN: hidden={} too small — limited expressiveness", hidden),
                LessonType::Warn
            ));
        }
    }

    // Analyze weight decay
    if let Some(wd) = config.weight_decay {
        if wd > 0.1 && rung.bpb > 3.0 {
            lessons.push((
                format!("AVOID: weight_decay={} too aggressive — causes underfitting", wd),
                LessonType::Avoid
            ));
        }
    }

    // Analyze optimizer
    if let Some(opt) = config.optimizer.as_deref() {
        if opt.contains("sgd") && rung.bpb > 2.7 {
            lessons.push((
                "PATTERN: SGD optimizer underperforms — use AdamW or Muon".to_string(),
                LessonType::Pattern
            ));
        }
    }

    // Early pruning pattern
    if rung.step <= 1000 && rung.bpb > 3.0 {
        lessons.push((
            format!("AVOID: BPB={} at rung-{} — early death, check config",
                     rung.bpb, rung.step),
            LessonType::Avoid
        ));
    }

    // If no specific lessons, generate generic one
    if lessons.is_empty() {
        lessons.push((
            format!("INFO: trial {} at BPB={}", outcome, rung.bpb),
            LessonType::Info
        ));
    }

    // Return most severe lesson
    let (lesson, lesson_type) = lessons.into_iter()
        .min_by(|a, b| {
            // Priority: Avoid > Pattern > Warn > Info > Winner
            let priority = |t: LessonType| match t {
                LessonType::Avoid => 0,
                LessonType::Pattern => 1,
                LessonType::Warn => 2,
                LessonType::Info => 3,
                LessonType::Winner => 4,
            };
            priority(a.1).cmp(&priority(b.1))
        })
        .unwrap();

    (lesson, lesson_type)
}

/// Store lesson in database
pub async fn store_lesson(
    db: &NeonDb,
    trial_id: &Uuid,
    outcome: Outcome,
    pruned_at_rung: i32,
    bpb_at_pruned: f64,
    lesson: &str,
    lesson_type: &str,
) -> Result<()> {
    db.store_lesson(
        trial_id,
        &outcome.to_string(),
        pruned_at_rung,
        bpb_at_pruned,
        lesson,
        lesson_type,
    ).await?;

    Ok(())
}

/// Get top lessons from experience
pub async fn get_top_lessons(
    db: &NeonDb,
    limit: i32,
) -> Result<Vec<(String, String, i32)>> {
    let lessons = db.get_top_lessons(limit).await?;
    Ok(lessons.into_iter().map(|l| (l.lesson, l.lesson_type, l.pattern_count)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lesson_high_lr() {
        let config = TrialConfig {
            lr: Some(0.07),
            d_model: Some(128),
            hidden: None,
            n_layers: None,
            optimizer: None,
            activation: None,
            weight_decay: None,
            dropout: None,
            warmup_steps: None,
            max_steps: None,
        };

        let rung = RungData { step: 1000, bpb: 3.4 };
        let (lesson, _lesson_type) = generate_lesson(&config, &rung, Outcome::Pruned);

        assert_eq!(_lesson_type, LessonType::Avoid);
        assert!(lesson.contains("lr=0.07"));
        assert!(lesson.contains("too high"));
    }

    #[test]
    fn test_lesson_small_model() {
        let config = TrialConfig {
            lr: Some(0.004),
            d_model: Some(64),
            hidden: None,
            n_layers: None,
            optimizer: None,
            activation: None,
            weight_decay: None,
            dropout: None,
            warmup_steps: None,
            max_steps: None,
        };

        let rung = RungData { step: 1000, bpb: 2.9 };
        let (lesson, _lesson_type) = generate_lesson(&config, &rung, Outcome::Pruned);

        assert!(lesson.contains("d_model=64"));
    }

    #[test]
    fn test_lesson_early_death() {
        let config = TrialConfig {
            lr: Some(0.01),
            d_model: Some(128),
            hidden: None,
            n_layers: None,
            optimizer: None,
            activation: None,
            weight_decay: None,
            dropout: None,
            warmup_steps: None,
            max_steps: None,
        };

        let rung = RungData { step: 1000, bpb: 3.2 };
        let (lesson, _lesson_type) = generate_lesson(&config, &rung, Outcome::Pruned);

        assert!(lesson.contains("BPB=3.2"));
        assert!(lesson.contains("rung-1000"));
    }

    #[test]
    fn test_lesson_type_priority() {
        let config = TrialConfig {
            lr: Some(0.07),
            d_model: Some(32),
            hidden: Some(16),
            n_layers: None,
            optimizer: None,
            activation: None,
            weight_decay: None,
            dropout: None,
            warmup_steps: None,
            max_steps: None,
        };

        let rung = RungData { step: 1000, bpb: 3.5 };
        let (_lesson, lesson_type) = generate_lesson(&config, &rung, Outcome::Pruned);

        // Should prioritize AVOID lessons
        assert_eq!(lesson_type, LessonType::Avoid);
    }
}
