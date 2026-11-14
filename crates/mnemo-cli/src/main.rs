//! Cli application

use std::io::Write;
use std::{
    path::{Path, PathBuf},
    pin::pin,
};

use clap::Args;
use clap::Parser;
use clap::Subcommand;
use futures::StreamExt;
use mnemo_core::answer::AskPlan;
use mnemo_core::context::Contextulizer;
use mnemo_core::ingest::{IngestionOptions, IngestionOutcome, ingest_document};
use mnemo_core::rerank::Reranker;
use mnemo_core::retrieve::{RetrieveParams, retrieve};
use mnemo_core::{
    answer::plan_answer,
    embed::Embedder,
    retrieve::{HybridParams, hybrid_search},
    store::Store,
};
use mnemo_eval::report::ModeMetrics;
use mnemo_llm::ollama::Ollama;
use tracing::{info, warn};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[derive(Parser, Debug)]
#[command(name = "mnemo", version = "0.0.1", about = "Offline data mining")]
struct Cli {
    /// Data store path
    #[arg(short, long)]
    data_dir: Option<PathBuf>,

    /// Operation command
    #[command(subcommand)]
    operation: Operation,
}
#[derive(Args, Debug)]
struct RetrievalOpts {
    /// No rerank
    #[arg(long)]
    no_rerank: bool,
    /// Rerank pool size.
    #[arg(long, default_value_t = 30)]
    rerank_pool: usize,
}

#[derive(Subcommand, Debug)]
enum Operation {
    /// Index the input file or files in the folder.
    Index {
        /// File or folder to index
        #[arg(short, long)]
        path: PathBuf,

        #[arg(short, long)]
        contextulize: bool,
    },
    /// Search the given query in the index and return matching
    Search {
        /// Search query
        #[arg(short, long)]
        query: String,

        /// Search options
        #[command(flatten)]
        opts: RetrievalOpts,
    },
    /// Ask a question and return answer with available indexed data
    Ask {
        /// User query
        #[arg(short, long)]
        question: String,
    },
    Eval {
        #[arg(long, default_value = "eval/dataset.jsonl")]
        path: PathBuf,
        #[arg(long, default_value_t = 8)]
        k: usize,

        #[arg(long, default_value = "eval/results.jsonl")]
        result: PathBuf,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // tracing_subscriber::registry()
    //     .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
    //     .with(fmt::layer())
    //     .init();
    let args = Cli::parse();

    let data_dir = match args.data_dir {
        Some(path) => path,
        None => {
            let path = Path::new("./data");
            path.to_path_buf()
        }
    };
    println!("Using data_dir {:?}", data_dir);
    match args.operation {
        Operation::Index { path, contextulize } => {
            let ctx = if contextulize {
                let llm = Ollama::new("http://localhost:11434", "qwen3.5:4b");
                llm.check_ready().await?;
                Some(&Contextulizer::new(llm))
            } else {
                None
            };
            let files = mnemo_core::load::get_files(&path)?;
            println!("files to be indexed {:?}", files);
            let bar = indicatif::ProgressBar::new(files.len() as u64);
            bar.set_style(indicatif::ProgressStyle::with_template(
                "{bar:40} {pos}/{len} docs  {elapsed_precise} elapsed  ~{eta} left  {msg}",
            )?);
            let store = Store::open(data_dir.join("db")).await?;
            let mut embedder = Embedder::new()?;
            let ingestopts = IngestionOptions {
                contextual: contextulize,
            };
            for file in files {
                let file_name = file.file_name().unwrap().to_string_lossy().to_string();
                bar.set_message(file_name);
                let ingestoutput =
                    ingest_document(&store, &mut embedder, ctx, file, &ingestopts).await?;
                match ingestoutput {
                    IngestionOutcome::Unchanged => info!("No change in indexing"),
                    IngestionOutcome::Indexed(cks) => info!("Indexed {cks} chunks"),
                }
                bar.inc(1);
            }
            bar.finish_with_message("done");
            //store.build_fts_context_index().await?;
            store.build_fts_index().await?;
        }
        Operation::Search { query, opts } => {
            println!("Peforming search with \"{}\"", query);
            let store = Store::open(data_dir.join("db")).await?;
            let mut embedder = Embedder::new()?;
            if opts.no_rerank {
                let p = HybridParams {
                    candidates: 5,
                    rrf_k: 60.0,
                    top_k: 3,
                };
                let retrieved = hybrid_search(&store, &mut embedder, &query, &p).await?;
                for h in retrieved.hits {
                    println!("Score {0}\n{1}\n\n", h.score, h.text);
                }
            } else {
                let p = RetrieveParams {
                    candidates: 5,
                    rerank_pool: opts.rerank_pool,
                    top_k: 3,
                    rrf_k: 60.0,
                    rerank: true,
                };
                let reranker = Reranker::new(data_dir.join("reranker"))?;
                let retrived = retrieve(&store, &mut embedder, &reranker, &query, &p).await?;
                for h in retrived.hits {
                    println!("Score {0}\n{1}\n\n", h.score, h.text);
                }
            }
        }
        Operation::Ask { question } => {
            println!("User asked \"{}\"", question);
            let ollama_cli = Ollama::new(
                "http://localhost:11434".to_string(),
                "qwen3.5:4b".to_string(),
            );
            ollama_cli.check_ready().await?;

            let p = HybridParams {
                candidates: 5,
                rrf_k: 60.0,
                top_k: 3,
            };
            let db = Store::open(data_dir.join("db")).await?;
            let mut embedder = Embedder::new()?;
            match plan_answer(&db, &mut embedder, &question, &p).await? {
                AskPlan::NoContext => warn!("I dont know based on the indexed document"),
                AskPlan::Ready {
                    messages,
                    sources: _,
                } => {
                    let resp = ollama_cli.chat_stream(&messages, 0.4, 8192).await?;
                    let mut resp = pin!(resp);

                    let mut answer = String::new();

                    while let Some(token) = resp.next().await {
                        let token = token?;
                        println!("{token}");
                        std::io::stdout().flush()?;
                        answer.push_str(&token);
                    }
                }
            }
        }
        Operation::Eval { path, k, result } => {
            let cases = mnemo_eval::dataset::load(&path)?;
            let reranker = Reranker::new(data_dir.join("models"))?;
            use mnemo_eval::runner::{Mode::*, run_mode};
            let mut embedder = Embedder::new()?;
            let store = Store::open(data_dir.join("db")).await?;
            for (name, mode) in [
                ("vector", Vector),
                ("fts", Fts),
                ("hybrid", Hybrid),
                ("hybridrerank", HybridRerank),
            ] {
                let (totals, per_query) =
                    run_mode(&mode, &cases, &store, &mut embedder, &reranker, k).await?;
                mnemo_eval::report::report(
                    &result,
                    &ModeMetrics {
                        mode: name.to_string(),
                        hit: totals.hit,
                        recall: totals.recall,
                        mrr: totals.mrr,
                        ndcg: totals.ndcg,
                        ms_per_query: per_query.as_millis() as f64,
                    },
                )?;
            }
        }
    }

    Ok(())
}
