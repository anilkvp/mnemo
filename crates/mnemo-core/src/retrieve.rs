/// Record redtrive from Lance db
use std::collections::HashMap;

use crate::embed::Embedder;

use crate::store::{Hits, Store};

pub struct HybridParams {
    pub candidates: usize,
    pub top_k: usize,
    pub rrf_k: f32,
}

pub struct Retrieved {
    pub hits: Vec<Hits>,
    pub best_distance: Option<f32>,
}

pub async fn hybrid_search(
    store: &Store,
    embedder: &mut Embedder,
    query: &str,
    p: &HybridParams,
) -> anyhow::Result<Retrieved> {
    let query_vector = embedder.embed_str(query)?;

    let (vec_hits, fts_hits) = tokio::try_join!(
        store.vector_search(&query_vector, p.candidates),
        store.fts_search(query, p.candidates)
    )?;
    let best_distance = vec_hits.first().map(|h| h.score);
    Ok(Retrieved {
        hits: rrf_fuse(&[vec_hits, fts_hits], p.rrf_k, p.top_k),
        best_distance,
    })
}

pub fn rrf_fuse(lists: &[Vec<Hits>], rrf_k: f32, top_k: usize) -> Vec<Hits> {
    let mut fused: HashMap<(String, u32), Hits> = HashMap::new();

    for list in lists {
        for (rank, hits) in list.iter().enumerate() {
            let contribusion = 1.0 / (rrf_k + rank as f32 + 1.0);
            fused
                .entry((hits.doc_id.clone(), hits.chunk_index))
                .and_modify(|h| h.score += contribusion)
                .or_insert_with(|| Hits {
                    score: contribusion,
                    ..hits.clone()
                });
        }
    }
    let mut out: Vec<Hits> = fused.into_values().collect();
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out.truncate(top_k);
    out
}
