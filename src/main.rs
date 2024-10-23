use crate::ai::candle::TextEmbedder;
use ftail::Ftail;
use log::LevelFilter;

mod ai;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    //create logs directory
    std::fs::create_dir_all("logs").unwrap();
    //init log
    Ftail::new()
        .console(LevelFilter::Debug)
        .daily_file("logs", LevelFilter::Error)
        .init()?;

    // //get current directory
    // let current_dir = std::env::current_dir().unwrap();
    // log::debug!("Current directory: {:?}", current_dir);

    // let binding = current_dir.join("models/bge-large-zh-v1.5");
    // let embd_model_path = binding.to_str().expect("Embedding model path not found");

    let embd_model_path = "/data/bge-large-zh-v1.5";
    // 初始化 TextEmbedder
    let embedder = TextEmbedder::new(embd_model_path, false, true, true)?;

    // 单个文本嵌入
    let text = "这是一个测试句子";
    let embedding = embedder.embed_text(text);
    match embedding {
        Ok(embedding) => {
            log::debug!("Single text embedding: {:?}", embedding);
            Ok(())
        }
        Err(e) => {
            log::error!("Error: {:?}", e);
            Err(e)
        }
    }
}
