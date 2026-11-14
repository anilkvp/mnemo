use std::{fs::OpenOptions, io::Write, path::PathBuf};

use serde::Serialize;

#[derive(Serialize)]
pub struct ModeMetrics {
    pub mode: String,
    pub hit: f64,
    pub recall: f64,
    pub mrr: f64,
    pub ndcg: f64,
    pub ms_per_query: f64,
}

pub fn report(path: &PathBuf, metric: &ModeMetrics) -> anyhow::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;

    let mut json_line = serde_json::to_string(&metric)?;
    json_line.push('\n');

    file.write_all(json_line.as_bytes())?;

    Ok(())
}
