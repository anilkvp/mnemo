use std::{path::PathBuf, sync::Mutex};

use fastembed::{RerankInitOptions, TextRerank};

pub struct Reranker {
    model: Mutex<TextRerank>,
}

impl Reranker {
    pub fn new(cache_dir: PathBuf) -> anyhow::Result<Self> {
        let model = TextRerank::try_new(
            RerankInitOptions::new(fastembed::RerankerModel::JINARerankerV1TurboEn)
                .with_cache_dir(cache_dir),
        )?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    pub fn score(&self, query: &str, docs: &[&str]) -> anyhow::Result<Vec<(usize, f32)>> {
        if docs.is_empty() {
            return Ok(vec![]);
        }
        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("lock is poisned"))?;
        let results = model.rerank(query, docs, false, Some(16))?;
        let mut scored: Vec<(usize, f32)> = results.iter().map(|r| (r.index, r.score)).collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(scored)
    }
}
