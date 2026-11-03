use std::fmt::Display;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DocId(pub u64);

impl Display for DocId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.to_string().as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: DocId,
    pub path: String,
    pub title: String,
    pub body: String,
    pub context_version: String,
    pub content_hash: String,
    pub chunk_count: usize,
    pub ingested_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: DocId,
    pub index: usize,
    pub title: String,
    pub text: String,
    pub path: String,
    pub context: Option<String>,
    pub embedding: Option<Vec<f32>>,
}
