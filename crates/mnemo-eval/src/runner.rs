use mnemo_core::{
    embed::Embedder,
    rerank::Reranker,
    retrieve::{self, RetrieveParams},
    store::Store,
};

use crate::{
    dataset::Case,
    metrics::{Totals, score},
    runner::Mode::HybridRerank,
};

pub enum Mode {
    Vector,
    Fts,
    Hybrid,
    HybridRerank,
}

pub async fn run_mode(
    mode: &Mode,
    cases: &[Case],
    store: &Store,
    embedder: &mut Embedder,
    reranker: &Reranker,
    k: usize,
) -> anyhow::Result<(Totals, std::time::Duration)> {
    let start = std::time::Instant::now();
    let mut t = Totals::default();

    for case in cases {
        let hits = match mode {
            Mode::Vector => {
                store
                    .vector_search(&embedder.embed_str(&case.query)?, k)
                    .await?
            }
            Mode::Fts => store.fts_search(&case.query, k).await?,
            Mode::Hybrid | Mode::HybridRerank => {
                let p = RetrieveParams {
                    rerank: matches!(mode, HybridRerank),
                    candidates: k,
                    top_k: k,
                    rerank_pool: k,
                    rrf_k: 60.0,
                };
                retrieve::retrieve(store, embedder, reranker, &case.query, &p)
                    .await?
                    .hits
            }
        };
        score(&hits, &case.relevant_set(), k, &mut t);
    }
    Ok((t, start.elapsed() / cases.len().max(1) as u32))
}
