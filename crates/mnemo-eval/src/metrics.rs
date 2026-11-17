use std::collections::HashSet;

use mnemo_core::store::Hit;
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

#[derive(Default, Debug)]
pub struct Totals {
    pub hit: f64,
    pub recall: f64,
    pub mrr: f64,
    pub ndcg: f64,
    pub n: usize,
}

impl Totals {
    pub fn averages(&self, mode: &str, ms_per_query: f64) -> ModeMetrics {
        let n = self.n.max(1) as f64;
        ModeMetrics {
            mode: mode.to_string(),
            hit: self.hit / n,
            recall: self.recall / n,
            mrr: self.mrr / n,
            ndcg: self.ndcg / n,
            ms_per_query,
        }
    }
}

pub fn score(hits: &[Hit], relevant: &HashSet<(String, u32)>, k: usize, t: &mut Totals) {
    let top = &hits[..hits.iter().len().min(k)];
    let is_rel = |h: &Hit| relevant.contains(&(h.title.to_string(), h.chunk_index));

    let found = top.iter().filter(|h| is_rel(h)).count();
    t.hit += (found > 0) as u8 as f64;
    t.recall += found as f64 / relevant.len() as f64;

    if let Some(pos) = top.iter().position(|h| is_rel(h)) {
        t.mrr += 1.0 / (pos as f64 + 1.0);
    }

    let dcg: f64 = top
        .iter()
        .enumerate()
        .filter(|(_, h)| is_rel(h))
        .map(|(i, _)| 1.0 / (i as f64 + 2.0).log2())
        .sum();

    let ideal: f64 = (0..relevant.len().min(k))
        .map(|i| 1.0 / (i as f64 + 2.0).log2())
        .sum();

    t.ndcg += if ideal > 0.0 { dcg / ideal } else { 0.0 };
    t.n += 1;
}

pub fn report(name: &str, t: &Totals, time_taken: std::time::Duration) {
    let n = t.n as f64;
    println!(
        "{name:<20} hit@10 {:.3} recall@10 {:.3} MRR {:.3} nDCG@10 {:.3} {:>3}ms",
        t.hit / n,
        t.recall / n,
        t.mrr / n,
        t.ndcg / n,
        time_taken.as_millis(),
    );
}
