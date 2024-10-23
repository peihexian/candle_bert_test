use anyhow::Result;
use candle_core::{backend::BackendDevice, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::Tokenizer;
pub struct TextEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    normalize_embeddings: bool,
    debug: bool,
}

impl TextEmbedder {
    pub fn new(
        model_id: &str,
        use_gpu: bool,
        normalize_embeddings: bool,
        debug: bool,
    ) -> Result<Self> {
        let start_time = chrono::Utc::now();
        let device = if use_gpu {
            Device::Cuda(candle_core::CudaDevice::new(0)?)
        } else {
            Device::Cpu
        };

        // 加载tokenizer
        let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_id))
            .map_err(|e| anyhow::anyhow!(e))?;

        // 加载模型配置
        let config_filename = format!("{}/config.json", model_id);
        let config_json_str = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config_json_str)?;

        // 加载模型权重
        let weights_path = format!("{}/pytorch_model.bin", model_id);
        let vb;

        vb = VarBuilder::from_pth(weights_path, DTYPE, &device)?; //&vec![weights_path]

        // 加载模型
        let model = BertModel::load(vb, &config)?;

        if debug {
            log::debug!("model_id: {}", model_id);
            log::debug!("use_gpu: {}", use_gpu);
            log::debug!("normalize_embeddings: {}", normalize_embeddings);
            log::debug!("debug: {}", debug);
            log::debug!("device: {:?}", device);
            let end_time = chrono::Utc::now();
            log::debug!(
                "load model cost: {} ms",
                end_time
                    .signed_duration_since(start_time)
                    .num_milliseconds()
            );
        }
        Ok(Self {
            model,
            tokenizer,
            device,
            normalize_embeddings,
            debug,
        })
    }

    pub fn embed_text(&self, text: &str) -> anyhow::Result<Tensor> {
        let start_time = chrono::Utc::now();
        // 对输入文本进行编码
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let input_ids = Tensor::new(encoding.get_ids(), &self.device)?;
        let token_type_ids = Tensor::new(encoding.get_type_ids(), &self.device)?;
        let attention_mask = Tensor::new(encoding.get_attention_mask(), &self.device)?;

        // 添加批次维度
        let input_ids = input_ids.unsqueeze(0)?;
        let token_type_ids = token_type_ids.unsqueeze(0)?;
        let attention_mask = attention_mask.unsqueeze(0)?;

        // 执行前向传播
        let output = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // 获取最后一层的隐藏状态
        let last_hidden_state = output;

        // 使用平均池化获取句子嵌入
        let mut embedding = last_hidden_state.mean(1)?;

        // 如果需要，进行L2归一化
        if self.normalize_embeddings {
            embedding = self.normalize_l2(&embedding)?;
        }
        if self.debug {
            let end_time = chrono::Utc::now();
            log::debug!(
                "embed_text cost: {} ms",
                end_time
                    .signed_duration_since(start_time)
                    .num_milliseconds()
            );
        }
        Ok(embedding)
    }

    pub fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Tensor> {
        let start_time = chrono::Utc::now();
        // 对输入文本进行编码
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // 构建输入张量
        let token_ids: Vec<Tensor> = encodings
            .iter()
            .map(|e| Tensor::new(e.get_ids(), &self.device).map_err(anyhow::Error::from))
            .collect::<Result<_>>()?;

        // 构建attention_mask张量
        let attention_masks: Vec<Tensor> = encodings
            .iter()
            .map(|e| Tensor::new(e.get_attention_mask(), &self.device).map_err(anyhow::Error::from))
            .collect::<Result<_>>()?;

        // 构建token_type_ids张量
        let token_type_ids: Vec<Tensor> = encodings
            .iter()
            .map(|e| Tensor::new(e.get_type_ids(), &self.device).map_err(anyhow::Error::from))
            .collect::<Result<_>>()?;

        // 堆叠张量
        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_masks, 0)?;
        let token_type_ids = Tensor::stack(&token_type_ids, 0)?;

        // 执行前向传播
        let output = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // 获取最后一层的隐藏状态
        let last_hidden_state = output;
        // 使用平均池化获取句子嵌入
        let mut embeddings = last_hidden_state.mean(1)?;

        // 如果需要，进行L2归一化
        if self.normalize_embeddings {
            embeddings = self.normalize_l2(&embeddings)?;
        }

        if self.debug {
            let end_time = chrono::Utc::now();
            log::debug!(
                "embed_batch cost: {} ms",
                end_time
                    .signed_duration_since(start_time)
                    .num_milliseconds()
            );
        }
        Ok(embeddings)
    }

    fn normalize_l2(&self, v: &Tensor) -> Result<Tensor> {
        Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }
}
