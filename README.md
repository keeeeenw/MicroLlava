# MicroLLaVA

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/keeeeenw/MicroLlava)

A compact vision language model that you can pretrain and finetune on a single consumer GPU such as NVIDIA RTX 4090 with 24GB VRAM.

## 📰 News and Updates

* 08/17/2025: the hugging face repo is renamed to https://huggingface.co/keeeeenw/MicroLlava.
* 08/17/2025: improved **VQAv2** average dev-test score from **44.01%** to **56.91%** by upgrading the vision tower from SigLip to SigLip2.
* 08/09/2025: initial version of MicroLlava released

## 🚀 Quick Start

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model from Hugging Face
hf_path = 'keeeeenw/MicroLlava'
model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
# model.cuda()  # Enable CUDA if needed - model runs fairly quickly on CPU

# Setup tokenizer
config = model.config
tokenizer = AutoTokenizer.from_pretrained(
    hf_path, 
    use_fast=False, 
    model_max_length=config.tokenizer_model_max_length,
    padding_side=config.tokenizer_padding_side
)

# Run inference
prompt = "What are the things I should be cautious about when I visit here?"
image_url = "https://llava-vl.github.io/static/images/view.jpg"

output_text, generation_time = model.chat(
    prompt=prompt,
    image=image_url,
    tokenizer=tokenizer
)

print(f'Model output: {output_text}')
print(f'Generation time: {generation_time}')
```

## 📋 Model Overview

| Component | Details |
|-----------|---------|
| **Framework** | Transformers + PyTorch |
| **Language Model** | [MicroLlama](https://huggingface.co/keeeeenw/MicroLlama) (~300M parameters) |
| **Vision Encoder** | [SigLIP2-SO400M](https://huggingface.co/google/siglip2-so400m-patch14-384) |
| **Training Hardware** | Single NVIDIA RTX 4090 |
| **Checkpoint Format** | SafeTensors |
| **License** | Apache 2.0 |

## 🎯 Key Features

- **🔧 Single GPU Training**: Train on consumer hardware without DeepSpeed
- **⚡ Fast Training**: Pretraining takes ~5 hours, finetuning ~12 hours on RTX 4090
- **📦 Compact**: Only ~300M language model parameters
- **🎨 Vision-Language Tasks**: Visual Question Answering, image captioning
- **🔄 Easy Iteration**: Perfect for research and experimentation

## 🏆 Performance

### VQAv2 Evaluation Results (MicroLlama 300M + Siglip2-so400m-patch4-384)

| Question Type | Accuracy |
|---------------|----------|
| Yes/No | 72.32% |
| Number | 43.89% |
| Other | 46.65% |
| **Overall** | **56.91%** |

*Evaluated on VQAv2 test-dev split*

### (Deprecated) VQAv2 Evaluation Results (MicroLlama 300M + Siglip-so400m-patch4-384)

| Question Type | Accuracy |
|---------------|----------|
| Yes/No | 65.08% |
| Number | 28.97% |
| Other | 29.32% |
| **Overall** | **44.01%** |

*Evaluated on VQAv2 test-dev split*

#### Planned tests include:
1. VQAv2 test set (instead of test-dev)
2. and datasets from TinyLlava evaluation
3. Community contributions with benchmark results are welcome and encouraged.

## 🛠️ Training

This model is based on [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) with optimizations for single GPU training.

### Training Times (RTX 4090)
- **Pretraining**: ~5 hours on LAION-CC-SBU-558K
- **Finetuning**: ~12 hours on TinyLLaVA datasets

### Key Training Modifications

**Pretraining Hyperparameters:**
- `gradient_accumulation_steps`: 2 → 8
- `learning_rate`: 1e-3 → 2.5e-4  
- `warmup_ratio`: 0.03 → 0.06
- `bfloat16`: True after the Siglip2 upgrade (improved stability)

**Finetuning:**
- Precision: `bfloat16` (improved stability)
- Same major hyperparameters as original TinyLLaVA

### Reproduce Training

1. Clone the training repository:
```bash
git clone https://github.com/keeeeenw/TinyLLaVA_Factory.git
cd TinyLLaVA_Factory
```
2. Follow the training guides in the repository for pretraining and finetuning steps.

## 🎯 Use Cases

### ✅ Intended Uses
- **Research**: Vision-language experimentation on limited hardware
- **Education**: Learning VLM concepts and implementations  
- **Prototyping**: Quick iteration for domain-specific applications
- **Finetuning**: Starting point for specialized vision-language tasks

### ⚠️ Limitations
- Small model size may limit complex reasoning capabilities
- OCR performance may be limited compared to larger models
- Performance varies with image quality and domain
- Minimal safety filtering - implement safeguards for production use

> **Warning**: This model should not be used for safety-critical applications without thorough human review and additional safeguards.

## 🔗 Related Projects

- [MicroLlama](https://huggingface.co/keeeeenw/MicroLlama) - The base language model
- [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) - Training framework
- [SigLIP2](https://huggingface.co/google/siglip2-so400m-patch14-384) - Vision encoder

## 📝 Citation

```bibtex
@misc{wang2024microllama,
  title        = {MicroLLaVA: a TinyLLaVA based VLM with MicroLlama 300M for single GPU training},
  author       = {Zixiao Ken Wang},
  year         = {2025},
  url          = {https://huggingface.co/keeeeenw/MicroLlava}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution
- Additional evaluation benchmarks
- Performance optimizations
- Documentation improvements
- Example applications

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to:
- [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) team for the training framework
- SigLIP2 authors for the efficient vision encoder
- LAION community for the pretraining datasets
- Hugging Face for model hosting and tools

---

⭐ **Star this repository if you find it useful!** ⭐

For questions and support, please open an issue or check out the [Hugging Face model page](https://huggingface.co/keeeeenw/MicroLlava-siglip-so400m-patch14-384-base-finetune).
