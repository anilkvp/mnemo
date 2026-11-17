/// Record redtrive from Lance db
use std::collections::HashMap;

use crate::embed::Embedder;

use crate::rerank::Reranker;
use crate::store::{Hit, Store};

pub struct HybridParams {
    pub candidates: usize,
    pub top_k: usize,
    pub rrf_k: f32,
}

pub struct Retrieved {
    pub hits: Vec<Hit>,
    pub best_distance: Option<f32>,
}

#[derive(Default, Debug)]
pub struct RetrieveParams {
    pub candidates: usize,  // per retrieve, e.g. 50
    pub rerank_pool: usize, // fused  candidates sent to reranker, e.g. 10
    pub top_k: usize,       // final e.g. 8
    pub rrf_k: f32,         // 60.0
    pub rerank: bool,       // toggle for A/B retrieve
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

pub fn rrf_fuse(lists: &[Vec<Hit>], rrf_k: f32, top_k: usize) -> Vec<Hit> {
    let mut fused: HashMap<(String, u32), Hit> = HashMap::new();

    for list in lists {
        for (rank, hits) in list.iter().enumerate() {
            let contribusion = 1.0 / (rrf_k + rank as f32 + 1.0);
            fused
                .entry((hits.doc_id.to_string(), hits.chunk_index))
                .and_modify(|h| h.score += contribusion)
                .or_insert_with(|| Hit {
                    score: contribusion,
                    ..hits.clone()
                });
        }
    }
    let mut out: Vec<Hit> = fused.into_values().collect();
    // out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.doc_id.0.cmp(&b.doc_id.0))
            .then_with(|| a.chunk_index.cmp(&b.chunk_index))
    });
    out.truncate(top_k);
    out
}

pub async fn retrieve(
    store: &Store,
    embedder: &mut Embedder,
    reranker: &Reranker,
    query: &str,
    p: &RetrieveParams,
) -> anyhow::Result<Retrieved> {
    let query_vector = embedder.embed_str(query)?;

    let (vec_hits, fts_hits) = tokio::try_join!(
        store.vector_search(&query_vector, p.candidates),
        store.fts_search(query, p.candidates)
    )?;

    let best_distance = vec_hits.first().map(|h| h.score);

    let pool_size = if p.rerank { p.rerank_pool } else { p.top_k };
    let fuzed = rrf_fuse(&[vec_hits, fts_hits], p.rrf_k, pool_size);

    let hits = if p.rerank {
        rerank_hits(reranker, query, fuzed, p.top_k)?
    } else {
        fuzed
    };
    Ok(Retrieved {
        hits,
        best_distance,
    })
}

fn rerank_hits(
    reranker: &Reranker,
    query: &str,
    pool: Vec<Hit>,
    top_k: usize,
) -> anyhow::Result<Vec<Hit>> {
    let docs: Vec<&str> = pool.iter().map(|f| f.text.as_str()).collect();
    let reranked = reranker.score(query, &docs)?;

    let mut slots: Vec<Option<Hit>> = pool.into_iter().map(Some).collect();

    Ok(reranked
        .into_iter()
        .take(top_k)
        .filter_map(|(i, score)| {
            slots[i].take().map(|mut h| {
                h.score = score;
                h.source = crate::store::RetrievalSource::Reranked;
                h
            })
        })
        .collect())
}
