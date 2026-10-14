//! Cli application

use std::io::Write;
use std::{
    path::{Path, PathBuf},
    pin::pin,
};

use clap::Parser;
use clap::Subcommand;
use futures::StreamExt;
use mnemo_core::answer::AskPlan;
use mnemo_core::{
    answer::plan_answer,
    embed::Embedder,
    ingest::load_and_chunk,
    retrieve::{HybridParams, hybrid_search},
    store::Store,
};
use mnemo_llm::ollama::Ollama;

#[derive(Parser, Debug)]
#[command(name = "mnemo", version = "0.0.1", about = "Offline data mining")]
struct Cli {
    #[arg(short, long)]
    data_dir: Option<PathBuf>,

    #[command(subcommand)]
    operation: Operation,
}

#[derive(Subcommand, Debug)]
enum Operation {
    Index {
        #[arg(short, long)]
        path: PathBuf,
    },
    Search {
        #[arg(short, long)]
        query: String,
    },
    Ask {
        #[arg(short, long)]
        question: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Cli::parse();

    let data_dir = match args.data_dir {
        Some(path) => path,
        None => {
            let path = Path::new("~/");
            path.to_path_buf()
        }
    };
    match &args.operation {
        Operation::Index { path } => {
            let chunks = load_and_chunk(path)?;
            let db = Store::open(data_dir.join("db")).await?;
            let status = db.upsert(&chunks).await?;
            db.build_fts_index().await?;
            println!("Upsert stats = {status:?}");
        }
        Operation::Search { query } => {
            let p = HybridParams {
                candidates: 5,
                rrf_k: 60.0,
                top_k: 3,
            };
            let db = Store::open(data_dir.join("db")).await?;
            let mut embedder = Embedder::new()?;
            let retrieved = hybrid_search(&db, &mut embedder, query, &p).await?;
            for h in retrieved.hits {
                println!("Score {0}\n{1}\n\n", h.score, h.text);
            }
        }
        Operation::Ask { question } => {
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
            match plan_answer(&db, &mut embedder, question, &p).await? {
                AskPlan::NoContext => println!("I dont know based on the indexed document"),
                AskPlan::Ready {
                    messages,
                    sources: _,
                } => {
                    let resp = ollama_cli.chat_stream(&messages, 0.4, 8192).await?;
                    let mut resp = pin!(resp);

                    let mut answer = String::new();

                    while let Some(token) = resp.next().await {
                        let token = token?;
                        print!("{token}");
                        std::io::stdout().flush()?;
                        answer.push_str(&token);
                    }
                }
            }
        }
    }

    Ok(())
}
