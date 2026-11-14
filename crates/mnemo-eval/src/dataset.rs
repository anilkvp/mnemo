use std::{collections::HashSet, path::Path};

use serde::Deserialize;

#[derive(Deserialize)]
pub struct Case {
    pub query: String,
    pub relevant: Vec<(String, u32)>,
    pub category: String,
}

impl Case {
    pub fn relevant_set(&self) -> HashSet<(String, u32)> {
        self.relevant.iter().cloned().collect()
    }
}

pub fn load(path: &Path) -> anyhow::Result<Vec<Case>> {
    std::fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| {
            serde_json::from_str(l).map_err(|e| anyhow::anyhow!("line {} error {}", (i + 1), e))
        })
        .collect()
}
