# **LoRA-based Instruction-tuned Language Model for Text Generation**

### **Project Overview**

This project leverages a LoRA (Low-Rank Adaptation) model for fine-tuning a pre-trained large language model (Qwen2.5-1.5B) to respond to task-specific instructions. The dataset used for training comes from the **LaMini-instruction** dataset, which contains a variety of task instructions paired with appropriate responses. The goal is to generate text based on instructions using a highly efficient LoRA-tuned model.

---

### **Installation and Setup**

To set up the project locally or in a Colab environment, follow these steps:

Install the required packages:

`!pip install -q datasets bitsandbytes peft pyarrow`

Ensure you have the correct dataset and model dependencies:

`from datasets import load_dataset`  
`from transformers import AutoTokenizer, AutoModelForCausalLM`  
`from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training`  
`import torch`  
---

### **Dataset**

The dataset is fetched from **LaMini-instruction** using Hugging Face's `datasets` library.

`dataset = load_dataset("MBZUAI/LaMini-instruction", split="train")`

A smaller subset is selected to perform fine-tuning.

`small_dataset = dataset.select(i for i in range(200))`  
---

### **Data Preprocessing**

We process the dataset to format instructions and responses:

1. **Prompt and Response Templates**:

Instructions are structured with a template and combined with responses.

A custom function `_add_text` adds new fields (`prompt`, `answer`, and `text`) to the dataset.

`prompt_template = """ Below is an instruction that describes a task, Write a response that appropriately completes the request, Instruction : {instruction} \n Response:"""`  
`answer_template = """{response}"""`

`def _add_text(rec):`  
    `rec["prompt"] = prompt_template.format(instruction=rec["instruction"])`  
    `rec["answer"] = answer_template.format(response=rec["response"])`  
    `rec["text"] = rec["prompt"] + rec["answer"]`  
    `return rec`

`small_dataset = small_dataset.map(_add_text)`

2. **Tokenization**:

The `tokenizer` converts the text data into tokenized form that can be processed by the model.

`tokenizer = AutoTokenizer.from_pretrained(model_id)`  
`tokenizer.pad_token = tokenizer.eos_token`

3. **Preprocessing Function**:

A partial function is created to preprocess the data using the tokenizer and truncate sequences longer than the max length.

`def _preprocess_batch(batch):`  
    `model_input = tokenizer(batch["text"], text_target=batch["response"], truncation=True, padding="max_length", max_length=256)`  
    `model_input["labels"] = model_input["input_ids"]`  
    `return model_input`  
---

### **Model Setup**

1. **Model Loading**:

A pre-trained large language model (`Qwen2.5-1.5B`) is loaded using the 

`AutoModelForCausalLM`.

`model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)`

2. **LoRA Configuration**:

LoRA (Low-Rank Adaptation) is used to fine-tune the model efficiently. This helps reduce the number of trainable parameters while maintaining the model's performance.

`lora_config = LoraConfig(r=256, lora_alpha=512, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM")`  
`model = get_peft_model(model, lora_config)`  
---

### **Training the Model**

1. **Training Arguments**:

The training configuration includes using mixed-precision training (`fp16`) and setting the number of epochs to 5\.

`training_args = TrainingArguments(`  
    `output_dir="Qwen2.5-1.5B-lora",`  
    `per_device_train_batch_size=1,`  
    `learning_rate=2e-5,`  
    `num_train_epochs=5,`  
    `logging_strategy="epoch",`  
    `evaluation_strategy="epoch"`  
`)`

2. **Trainer**:

A `Trainer` object from Hugging Face is used to manage the training and evaluation processes.

`trainer = Trainer(`  
    `model=model,`  
    `tokenizer=tokenizer,`  
    `args=training_args,`  
    `data_collator=data_collator,`  
    `train_dataset=split_dataset["train"],`  
    `eval_dataset=split_dataset["test"]`  
`)`  
---

### **Postprocessing and Inference**

After training, the model is used to generate responses to new prompts.

1. **Inference Pipeline**:

A text-generation pipeline is created for inference.

`inf_pipeline = pipeline("text-generation", model=trainer.model, tokenizer=tokenizer, max_length=256)`

2. **Postprocessing the Response**:

A simple function `post process` ensures that the model's response matches the expected format.

`def postprocess(response):`  
    `message = response.split("Response:")`  
    `return "".join(message[1:])`

3. **Example Usage**:

A sample instruction ("Write me a recipe for Dosa") is passed, and the generated response is displayed.

`inference_prompt = "Write me a recipe for Dosa"`  
`response = inf_pipeline(prompt_template.format(instruction=inference_prompt))[0]["generated_text"]`  
`formatted_response = postprocess(response)`  
`print(formatted_response)`  
---

### **Output**

The model successfully generates a recipe for "Dosa" based on the provided instruction prompt. However, an error about unsupported models for text-generation is encountered due to the `PeftModelForCausalLM`.

---

### **Dependencies**

* `datasets`  
* `transformers`  
* `peft`  
* `bitsandbytes`  
* `torch`  
* `pyarrow`

---

### **Future Improvements**

1. **Resolve Model Compatibility**: Modify the pipeline for better compatibility with PEFT models or switch to a supported model for causal text generation.  
2. **Experiment with Hyperparameters**: Explore changes in LoRA configurations and learning rates to optimize performance.  
3. **Increase Dataset Size**: Test the model with larger datasets for better instruction-based learning.

