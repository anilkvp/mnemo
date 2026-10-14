//

use mnemo_llm::ollama::Message;

use crate::{
    embed::Embedder,
    retrieve::{HybridParams, hybrid_search},
    store::{Hits, Store},
};

const MAX_DISTANCE: f32 = 0.45;

const SYSTEM_PROMPT: &str = "\
You answer questions using ONLY the numbered sources provided.
Rules:
    - Cite every claim with its source number in square brackets, e.g. [1] or [2][3].
    - If the sources to not contain the answer, reply exactly: I don't know based on the indexed documents
    - Do not use outside knowledge. Be concise.
";

pub fn build_messages(question: &str, hits: &[Hits]) -> Vec<Message> {
    let mut sources = String::new();
    for (i, hit) in hits.iter().enumerate() {
        sources.push_str(&format!("[{}] from {} \n{}\n\n", i + 1, hit.path, hit.text));
    }
    vec![
        Message::system(SYSTEM_PROMPT),
        Message::user(format!("Sources:\n\n{sources}\n\nQuestion:{question}")),
    ]
}

pub enum AskPlan {
    NoContext,
    Ready {
        messages: Vec<Message>,
        sources: Vec<Hits>,
    },
}

pub async fn plan_answer(
    store: &Store,
    embedder: &mut Embedder,
    question: &str,
    params: &HybridParams,
) -> anyhow::Result<AskPlan> {
    let retrived = hybrid_search(store, embedder, question, params).await?;
    match retrived.best_distance {
        Some(d) if d <= MAX_DISTANCE && !retrived.hits.is_empty() => Ok(AskPlan::Ready {
            messages: build_messages(question, &retrived.hits),
            sources: retrived.hits,
        }),
        _ => Ok(AskPlan::NoContext),
    }
}
