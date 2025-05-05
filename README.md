# Qwen Arabic Fine-tuning Project

This project fine-tunes the Qwen2-1.5B model for Arabic language tasks using Quantized LoRA (QLoRA). [Paper Link](https://doi.org/10.1007/978-3-031-83793-7_27)



# Qwen-Arabic Evaluation on ArabicMMLU

Eevaluation of the Qwen-Arabic language model (1.5B parameters) on the ArabicMMLU benchmark. The model demonstrates strong parameter efficiency while maintaining competitive performance across various knowledge domains.

## Model Overview

Qwen-Arabic is a 1.5B parameter language model fine-tuned for Arabic language tasks. It is based on the Qwen architecture and optimized using QLoRA (Quantized Low-Rank Adaptation) techniques.

## Performance Results

### Overall Performance
- Average Accuracy: 42.3%
- Best Category: Social Science (46.1%)
- Most Challenging: Arabic Language (37.8%)

### Category-wise Performance
| Category         | Accuracy (%) |
|-----------------|--------------|
| STEM            | 42.2         |
| Social Science  | 46.1         |
| Humanities      | 41.8         |
| Arabic Language | 37.8         |
| Other           | 42.9         |
| Average         | 42.3         |

### Efficiency Analysis
- Performance per Billion Parameters: 28.20 accuracy points
- 389.0x more parameter-efficient than GPT-4
- Achieves 58.3% of GPT-4's performance with only 0.15% of parameters

### Comparison with Other Models
| Model              | Parameters | Average Accuracy | Efficiency Score |
|-------------------|------------|------------------|------------------|
| GPT-4             | ~1000B     | 72.5%           | 0.072           |
| Jais-chat         | 30B        | 62.3%           | 2.077           |
| AceGPT-chat       | 13B        | 52.6%           | 4.046           |
| Qwen-Arabic       | 1.5B       | 42.3%           | 28.200          |


## Prerequisites

- Ubuntu (or similar Linux distribution)
- Python 3.10
- CUDA-compatible GPU with at least 4GB VRAM
- At least 12GB system RAM
- Ollama installed and configured

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/prakash-aryan/qwen-arabic-project.git
   cd qwen-arabic-project
   ```

2. Create and activate a virtual environment:
   ```
   python3.10 -m venv qwen_env
   source qwen_env/bin/activate
   ```

3. Install the required packages:
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Install PyTorch with CUDA support:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Project Structure

```
qwen-arabic-project/
├── data/
│   └── arabic_instruction_dataset/
├── models/
├── results/
├── src/
│   ├── compare_qwen_models.py
│   ├── evaluate_arabic_model.py
│   ├── finetune_qwen.py
│   ├── get_datasets.py
│   ├── load_and_merge_model.py
│   ├── preprocess_datasets.py
│   └── validate_dataset.py
├── tools/
│   └── llama-quantize
├── requirements.txt
├── run_pipeline.sh
├── Modelfile
└── README.md
```

## Usage

1. Download and prepare datasets:
   ```
   python src/get_datasets.py
   ```

2. Preprocess and combine datasets:
   ```
   python src/preprocess_datasets.py
   ```

3. Validate the dataset:
   ```
   python src/validate_dataset.py
   ```

4. Fine-tune the model:
   ```
   python src/finetune_qwen.py --data_path ./data/arabic_instruction_dataset --output_dir ./models/qwen2_arabic_finetuned --num_epochs 3 --batch_size 1 --gradient_accumulation_steps 16 --learning_rate 2e-5
   ```

5. Load and merge the fine-tuned model:
   ```
   python src/load_and_merge_model.py
   ```

6. Convert to GGUF format:
   ```
   python src/convert_hf_to_gguf.py ./models/qwen2_arabic_merged_full --outfile ./models/qwen_arabic_merged_full.gguf
   ```

7. Quantize the model:
   ```
   ./tools/llama-quantize ./models/qwen_arabic_merged_full.gguf ./models/qwen_arabic_merged_full_q4_k_m.gguf q4_k_m
   ```

8. Create Ollama model:
   ```
   ollama create qwen-arabic-custom -f Modelfile
   ```

9. Evaluate the model:
   ```
   python src/evaluate_arabic_model.py
   ```

10. Compare models:
    ```
    python src/compare_qwen_models.py
    ```

## Running the Full Pipeline

To run the entire pipeline from data preparation to model evaluation, use the provided shell script:

```
chmod +x run_pipeline.sh
./run_pipeline.sh
```

## Notes

- Ensure you have sufficient disk space for the datasets and model files.
- The fine-tuning process can take several hours to days, depending on your hardware.
- Monitor GPU memory usage during fine-tuning and adjust batch size or gradient accumulation steps if necessary.
- Make sure to have Ollama installed for the model creation and evaluation steps.

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try reducing the batch size or increasing gradient accumulation steps.
- For any other issues, please check the error logs or open an issue in the GitHub repository.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

This means:
- You can use, modify, and distribute this software.
- If you distribute modified versions, you must also distribute them under the GPL-3.0.
- You must include the original copyright notice and the license text.
- You must disclose your source code when you distribute the software.
- There's no warranty for this free software.

For more details, see the [LICENSE](LICENSE) file in this repository or visit [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Acknowledgements

This project uses the following main libraries and tools:
- Transformers by Hugging Face
- PyTorch
- PEFT (Parameter-Efficient Fine-Tuning)
- Ollama
- GGUF (for model conversion)


Here's the markdown code specifically for the citation section that you can add to your README.md file:

## Citation

If you use this work in your research, please cite:

```bibtex
@InProceedings{10.1007/978-3-031-83793-7_27,
  author="Aryan, Prakash",
  editor="Verma, Anshul and Verma, Pradeepika and Pattanaik, Kiran Kumar and Buyya, Rajkumar and Dasgupta, Dipankar",
  title="Resource-Aware Arabic LLM Creation: Model Adaptation, Integration, and Multi-domain Testing",
  booktitle="Advanced Network Technologies and Intelligent Computing",
  year="2025",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="415--434",
  abstract="This paper presents a novel approach to fine-tuning the Qwen2-1.5B model for Arabic language processing using Quantized Low-Rank Adaptation (QLoRA) on a system with only 4 GB VRAM. We detail the process of adapting this large language model to the Arabic domain, using diverse datasets including Bactrian, OpenAssistant, and Wikipedia Arabic corpora. Our methodology involves custom data preprocessing, model configuration, and training optimization techniques such as gradient accumulation and mixed-precision training. We address specific challenges in Arabic NLP, including morphological complexity, dialectal variations, and diacritical mark handling. Experimental results over 10,000 training steps show significant performance improvements, with the final loss converging to 0.1083. We provide comprehensive analysis of GPU memory usage, training dynamics, and model evaluation across various Arabic language tasks, including text classification, question answering, and dialect identification. The fine-tuned model demonstrates robustness to input perturbations and improved handling of Arabic-specific linguistic phenomena. This research contributes to multilingual AI by demonstrating a resource-efficient approach for creating specialized language models, potentially democratizing access to advanced NLP technologies for diverse linguistic communities. Our work paves the way for future research in low-resource language adaptation and efficient fine-tuning of large language models.",
  isbn="978-3-031-83793-7"
}
```

**Paper Reference:**
Aryan, P. (2025). Resource-Aware Arabic LLM Creation: Model Adaptation, Integration, and Multi-domain Testing. In: Verma, A., Verma, P., Pattanaik, K.K., Buyya, R., Dasgupta, D. (eds) Advanced Network Technologies and Intelligent Computing. ANTIC 2024. Communications in Computer and Information Science, vol 2335. Springer, Cham. https://doi.org/10.1007/978-3-031-83793-7_27

**DOI:** https://doi.org/10.1007/978-3-031-83793-7_27  
**Published:** 08 March 2025  
**Publisher:** Springer, Cham  
**Print ISBN:** 978-3-031-83792-0  
**Online ISBN:** 978-3-031-83793-7
