//! This is an example project.

use llama_cpp_2::context;
use llama_cpp_2::context::sample::sampler::{SampleStep, Sampler};
use llama_cpp_2::model::AddBos;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::{context::params::LlamaContextParams, model::LlamaModel};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use anyhow::{Context, Result};

use std::{num::NonZeroU32, path::Path};

fn main() -> Result<()> {
    let backend = LlamaBackend::init()?;

    // TODO: Performance: Implement NUMA init -> llama_numa_init
    // https://github.com/ggerganov/llama.cpp/blob/bfe7dafc9cf96b9a09ead347fed9a547930fc631/examples/main/README.md?plain=1#L285
    
    let prompt_path = Path::new("./prompt");
    let prompt = std::fs::read_to_string(prompt_path)
        .with_context(|| format!("failed to read prompt"))?;

    let model_params = {
        // #[cfg(feature = "cublas")]
        // if !disable_gpu {
        //     LlamaModelParams::default().with_n_gpu_layers(1000)
        // } else {
        //     LlamaModelParams::default()
        // }
        // #[cfg(not(feature = "cublas"))]
        LlamaModelParams::default()
    };

    // initialize the context
    let ctx_params = LlamaContextParams::default();

    let model_path = Path::new("./llama-2-7b.Q4_K_M.gguf");

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .with_context(|| "unable to load model")?;
    let context = model
        .new_context(&backend, ctx_params.clone())
        .with_context(|| "unable to create the llama_context")?;

    let training_ctx = model.n_ctx_train();
    let n_ctx = ctx_params.n_ctx().unwrap().get();

    if n_ctx > training_ctx {
        eprintln!("Warning: The model has a maximum context of {} while the context is set to {}", training_ctx, n_ctx);
    }

    let prompt_pfx = "\n\n### Instruction:\n\n";
    let prompt_sfx = "\n\n### Response:\n\n";
    let full_prompt: String = String::from(prompt_pfx) + &prompt + &String::from(prompt_sfx);

    // BOS token -> Models typically perform better when tokens they were trained with are prepended to the input
    let tokens_list = model
        .str_to_token(&full_prompt, AddBos::Always) 
        .with_context(|| format!("failed to tokenize {prompt}"))?;

    //From output which was decent:
    //generate: n_ctx = 512, n_batch = 2048, n_predict = 128, n_keep = 1

    // new tokens to predict @default -1
    let n_predit: i32 = 128;
    // number of tokens to keep from initial prompt @default 0
    let n_keep: i32 = 1;

    let top_k = 40;
    let tail_free = 1.0;
    let typical = 1.0;
    let top_p = 0.95;
    let min_p = 0.05;
    let temp = 0.8;
    // number of previous tokens to remember
    let n_prev= 64.0;
    // if greater than 0, output the probabilities of top n_probs tokens. 
    let n_probs = 0;



    let finalizer = &|mut canidates: LlamaTokenDataArray, history: &mut Vec<LlamaToken>| {
        canidates.sample_softmax(None);
        let token = canidates.data[0];
        history.push(token.id());
        vec![token]
    };
    let mut history: Vec<LlamaToken> = vec![];
    let mut sampler = Sampler::new(finalizer);
    
    let repeat_penalty_name = String::from("Repeat Penalty");
    let top_k_name = String::from("Top K");
    let tail_free_name = String::from("TailFree");
    let typical_name = String::from("Typical");
    let top_p_name = String::from("Top P");
    let min_p_name = String::from("Min P");
    let temp_name = String::from("Temp");
    let repeat_penalty_step = SampleStep::new(repeat_penalty_name, Box::new(|c: &mut LlamaTokenDataArray, history: &mut Vec<LlamaToken>| c.sample_repetition_penalty(None, history, 64, 1.1, 0.0, 0.0)));
    let top_k_step = SampleStep::new(top_k_name, Box::new(|c: &mut LlamaTokenDataArray, _: &mut Vec<LlamaToken>| c.sample_top_k(None, 40, 1)));
    let tail_free_step = SampleStep::new(tail_free_name, Box::new(|c: &mut LlamaTokenDataArray, _: &mut Vec<LlamaToken>| c.sample_tail_free(None, 1.0, 1)));
    let typical_step = SampleStep::new(typical_name, Box::new(|c: &mut LlamaTokenDataArray, _: &mut Vec<LlamaToken>| c.sample_typical(None, 1.0, 1)));
    let top_p_step = SampleStep::new(top_p_name, Box::new(|c: &mut LlamaTokenDataArray, _: &mut Vec<LlamaToken>| c.sample_top_p(None, 0.95, 1)));
    let min_p_step = SampleStep::new(min_p_name, Box::new(|c: &mut LlamaTokenDataArray, _: &mut Vec<LlamaToken>| c.sample_min_p(None, 0.05, 1)));
    let temp_step = SampleStep::new(temp_name, Box::new(|c: &mut LlamaTokenDataArray, _: &mut Vec<LlamaToken>| c.sample_temp(None, 0.5)));

    sampler.push_step(&repeat_penalty_step);
    sampler.push_step(&top_k_step);
    sampler.push_step(&tail_free_step);
    sampler.push_step(&typical_step);
    sampler.push_step(&top_p_step);
    sampler.push_step(&min_p_step);
    sampler.push_step(&temp_step);

    fn print_sampling_order(sampler: &Sampler<Vec<LlamaToken>>) {
        let mut order = String::new();
        for step in sampler.steps.iter() {
            order.push_str(&format!("{} -> ", step.name));
        }
        order.push_str("finalizer");
        println!("Sampling order: {}", order);
    }

    // println!("Sampling: {}", )
    // LOG_TEE("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
    print_sampling_order(&sampler);
    println!("generate: n_ctx = {}, n_batch = {}, n_predict = {}, n_keep = {}", n_ctx, ctx_params.n_batch(), n_predit, n_keep);
    // TODO: Support CFG

    Ok(())
}