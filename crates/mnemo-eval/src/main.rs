use std::path::Path;

use mnemo_core::{embed::Embedder, rerank::Reranker, store::Store};
use mnemo_eval::{
    dataset,
    history::{self, RunRecord},
    runner::{
        Mode::{Fts, Hybrid, HybridRerank, Vector},
        run_mode,
    },
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let k = 3;
    let eval_dataset_path = Path::new("eval/dataset.jsonl");
    let cases = dataset::load(eval_dataset_path)?;

    println!("Found {} cases to tests", cases.len());

    let base_location = Path::new("data");
    let reranker = Reranker::new(base_location.join("models"))?;
    let mut embedder = Embedder::new()?;
    let store = Store::open(base_location.join("db")).await?;
    let mut metrics = Vec::new();
    for (name, mode) in [
        ("vector", Vector),
        ("fts", Fts),
        ("hybrid", Hybrid),
        ("hybridrerank", HybridRerank),
    ] {
        let (totals, per_query) =
            run_mode(&mode, &cases, &store, &mut embedder, &reranker, k).await?;
        metrics.push(totals.averages(name, per_query.as_millis() as f64));
        println!("mode: {}, totals {:?}", name, totals);
    }
    history::append(
        Path::new("eval/results.jsonl"),
        RunRecord {
            timestamp: chrono::Utc::now().timestamp(),
            git_commit: history::git_commit(),
            release: true,
            n_cases: cases.len(),
            notes: Some("baseline deterministic, repeat".to_string()),
            modes: metrics,
        },
    )?;
    Ok(())
}
