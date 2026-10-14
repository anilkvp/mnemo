// Text embedding model
use std::sync::Mutex;

use fastembed::{TextEmbedding, TextInitOptions};

pub struct Embedder {
    model: Mutex<TextEmbedding>,
}

impl Embedder {
    pub fn new() -> anyhow::Result<Embedder> {
        let model = TextEmbedding::try_new(
            TextInitOptions::new(fastembed::EmbeddingModel::BGESmallENV15)
                .with_show_download_progress(true),
        )?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    pub fn embed(&self, texts: &Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut emb_model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("embedding model lock poised"))?;
        let embeddings = emb_model.embed(texts, None)?;
        Ok(embeddings)
    }

    pub fn embed_str(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut emb_model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("embedding model lock poised"))?;
        let mut embeddings = emb_model.embed([text.to_string()], None)?;
        embeddings
            .pop()
            .ok_or_else(|| anyhow::anyhow!("embedding model return no vector"))
    }
}
