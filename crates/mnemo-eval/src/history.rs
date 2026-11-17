use std::io::Write;
use std::path::Path;

use serde::Serialize;

use crate::metrics::ModeMetrics;

#[derive(Serialize)]
pub struct RunRecord {
    pub timestamp: i64,
    pub git_commit: String,
    pub release: bool,
    pub n_cases: usize,
    pub notes: Option<String>,
    pub modes: Vec<ModeMetrics>,
}

pub fn append(path: &Path, record: RunRecord) -> anyhow::Result<()> {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{}", serde_json::to_string(&record)?)?;

    Ok(())
}

pub fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default()
}
