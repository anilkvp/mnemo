use crate::types::{Chunk, Document};
use mnemo_llm::ollama::{Message, Ollama};
use tracing::error;

const MAX_CONTEXT_TOKENS: u32 = 100;
const MAX_DOC_CHAR: usize = 12_0000;
const NEIGHBOURS: usize = 2;

const SYSTEM: &str = "\
You write short context for search indexing. Given a document and one chunk from it, \
write 1-2 sentences that situate the chunk within the document: what document it is from, \
what topic or section it belongs to, and what any pronouns or vague references refer to. \
Output only the context sentences, nothing else.";

#[derive(Default, Debug)]
pub struct ContextStats {
    pub ok: usize,
    pub empty: usize,
    pub failed: usize,
}

pub struct Contextulizer {
    llm: Ollama,
}

impl Contextulizer {
    pub fn new(llm: Ollama) -> Self {
        Self { llm }
    }

    pub async fn add_context(&self, doc: &Document, chunks: &mut [Chunk]) -> ContextStats {
        let mut stats = ContextStats::default();
        let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        for (i, chunk) in chunks.iter_mut().enumerate() {
            let source = document_view(doc, &chunk_texts, i);
            let messages = vec![
                Message::system(SYSTEM),
                Message::user(format!(
                    "<document title=\"{}\">\n{source}\n</document>\n\n<chunk>\n{}\n</chunk>",
                    chunk.title, chunk.text
                )),
            ];

            match self.llm.complete(&messages, MAX_CONTEXT_TOKENS).await {
                Ok(raw) => match clean_context(&raw) {
                    Some(context) => {
                        chunk.context = Some(context);
                        stats.ok += 1;
                    }
                    None => {
                        stats.empty += 1;
                    }
                },
                Err(e) => {
                    error!("Failed in adding context {e}");
                    stats.failed += 1;
                }
            }
        }
        stats
    }

    pub fn model_name(&self) -> String {
        self.llm.model_name()
    }
}

fn document_view(doc: &Document, chunk_texts: &[String], i: usize) -> String {
    if doc.body.len() >= MAX_DOC_CHAR {
        return doc.body.clone();
    }

    let intro: String = doc.body.chars().take(1_500).collect();
    let lo = i.saturating_sub(NEIGHBOURS);
    let hi = (i + NEIGHBOURS + 1).min(chunk_texts.len());

    format!("{intro}\n\n[...]\n\n{}", chunk_texts[lo..hi].join("/n/n"))
}

fn clean_context(raw: &str) -> Option<String> {
    let mut s = raw.trim();
    for prefix in ["Context", "Here in the context", "Here is the context"] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest.trim();
        }
    }
    let s = s.trim_matches('"').trim();
    if s.is_empty() {
        return None;
    }
    Some(s.chars().take(400).collect())
}
