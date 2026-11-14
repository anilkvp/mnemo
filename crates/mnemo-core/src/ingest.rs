/// Ingetion related functions
use crate::{
    context::Contextulizer,
    embed::Embedder,
    store::Store,
    types::{Chunk, DocId, Document},
};
use std::{
    fs,
    hash::{Hash, Hasher},
    path::PathBuf,
};
use text_splitter::{ChunkConfig, TextSplitter};

const CONTEXT_VERSIION: usize = 1;

pub enum IngestionOutcome {
    Unchanged,
    Indexed(usize),
}

pub struct IngestionOptions {
    pub contextual: bool,
}

pub async fn ingest_document(
    store: &Store,
    embedder: &mut Embedder,
    ctx: Option<&Contextulizer>,
    file: PathBuf,
    opts: &IngestionOptions,
) -> anyhow::Result<IngestionOutcome> {
    let doc = load(&file)?;
    let hash = content_hash(&doc.body);
    let version = content_version(opts, ctx);

    if store.document_state(doc.id).await? == Some((hash.clone(), version.clone())) {
        return Ok(IngestionOutcome::Unchanged);
    }
    let mut chunks = chunk_document(&doc)?;

    if let (true, Some(c)) = (opts.contextual, ctx) {
        let stats = c.add_context(&doc, &mut chunks).await;
        tracing::info!(doc=%doc.path, ?stats, "contextualized");
    }

    embedder.embed_chunks(&mut chunks)?;

    store.delete_chunks_for(&doc.id).await?;
    store.insert_chunks(&chunks).await?;
    store
        .upsert_document(&doc, hash, version, chunks.len())
        .await?;
    Ok(IngestionOutcome::Indexed(chunks.len()))
}

fn chunk_document(doc: &Document) -> anyhow::Result<Vec<Chunk>> {
    let text_splitter = TextSplitter::new(ChunkConfig::new(1000).with_overlap(50)?);
    Ok(text_splitter
        .chunks(&doc.body)
        .enumerate()
        .map(|(i, c)| Chunk {
            id: doc.id,
            index: i,
            title: doc.title.clone(),
            path: doc.path.clone(),
            text: c.to_string(),
            context: None,
            embedding: None,
        })
        .collect())
}

fn load(path: &PathBuf) -> anyhow::Result<Document> {
    let docid = doc_id_from_path(path);
    let title = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    let content = fs::read_to_string(path.clone())?;
    let content_hash = content_hash(&content);
    Ok(Document {
        id: docid,
        chunk_count: 0,
        path: path.to_string_lossy().to_string(),
        context_version: "1v:Document".to_owned(),
        ingested_at: 345,
        body: content,
        title,
        content_hash,
    })
}

fn doc_id_from_path(path: &PathBuf) -> DocId {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut h);
    DocId(h.finish())
}

fn content_hash(content: &str) -> String {
    blake3::hash(content.as_bytes()).to_hex().to_string()
}

fn content_version(opts: &IngestionOptions, ctx: Option<&Contextulizer>) -> String {
    match (opts.contextual, ctx) {
        (true, Some(c)) => format!("v{CONTEXT_VERSIION}:{}", c.model_name()),
        _ => "none".to_string(),
    }
}
