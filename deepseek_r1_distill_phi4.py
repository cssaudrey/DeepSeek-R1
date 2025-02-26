
# 基于deepseek-r1范式的phi4蒸馏模型
import os, sys
from dotenv import load_dotenv
load_dotenv()

OUTPUT_DIR = "./chkp_dir/deepseek_r1_distill_phi4/"
import torch
from unsloth import FastLanguageModel
# 从FastLanguageModel类中加载预训练的语言模型和对应的分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "microsoft/phi-4",  # 指定预训练模型的名称，这里使用的是微软的phi-4模型
    device_map='cuda:0',  # 指定模型加载到哪个设备上，这里是第一个CUDA设备（通常是GPU）
    trust_remote_code = True,  # 允许从远程服务器加载模型代码，即使代码是未经验证的
    attn_implementation="flash_attention_2",  # 指定注意力机制的实现方式，这里使用的是flash_attention_2
)


# 获取一个PEFT（Parameter-Efficient Fine-Tuning）模型，用于高效微调
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, #控制微调过程。建议：8、16、32、64 或 128。更高：在困难任务上的准确性更高，但会增加内存和过拟合的风险。
    lora_alpha = 16, # 缩放因子：确定学习强度。建议：等于或加倍秩 
    lora_dropout = 0, #正则化的 dropout 概率。
    bias = "none",    
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context。减少长上下文的内存使用量
    random_state = 3407, # seed
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
)

# 从unsloth.chat_templates模块导入get_chat_template函数
from unsloth.chat_templates import get_chat_template

# 调用get_chat_template函数，传入tokenizer和chat_template参数
# chat_template: 指定使用的聊天模板，这里设置为"phi-4"
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "phi-4",
)

from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from datasets import load_dataset

# 使用社区中用DeepSeek生成的数据，
dataset = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k")

# Olivia选择前1000条样本测试
dataset['train'] = dataset['train'].select(range(1000))
dataset['test'] = dataset['test'].select(range(90)) 

trainer = SFTTrainer(
    model = model,  
    tokenizer = tokenizer,
    train_dataset = dataset['train'],  # 训练数据集
    eval_dataset=dataset['test'],  # 测试数据集
    max_seq_length = 5120,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),  # 数据整理器，用于序列到序列的任务
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    # 创建一个SFTConfig对象，用于配置训练过程中的各种参数
    args = SFTConfig(
        log_level = "info",  # 设置日志级别为信息级别
        logging_strategy = "steps",
        logging_steps = 5,  # 每5步记录一次日志
        output_dir = OUTPUT_DIR,
        overwrite_output_dir = True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps = 100,# 每100步进行一次评估
        per_device_eval_batch_size=8,  # 每个设备的评估批次大小为8
        save_strategy = "steps", #"best",
        save_steps = 100,
        save_total_limit = 8,
        per_device_train_batch_size = 8,# 
        gradient_accumulation_steps =4 ,#  
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False}, 
        num_train_epochs = 3, #1
        max_steps = -1,
        learning_rate = 2e-4, #学习率
        fp16 = not is_bfloat16_supported(),  # 检查是否支持bfloat16（bf16）格式,如果支持bfloat16，则将fp16设置为False，因为fp16和bf16不能同时使用
        bf16 = is_bfloat16_supported(),  # 如果支持bfloat16，则将bf16设置为True
        lr_scheduler_type = "cosine", # linear
        seed = 3407,
        report_to = "none", # Use this for WandB etc,
        remove_unused_columns = True,
    ),
)

from unsloth.chat_templates import train_on_responses_only  
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user<|im_sep|>",
    response_part="<|im_start|>assistant<|im_sep|>",
)

##########模型训练#########
trainer_stats = trainer.train()  

# 训练完成后，获取训练过程中的各种指标：

# 从trainer_stats对象中获取训练过程中的度量指标
metrics = trainer_stats.metrics
# 使用trainer对象的log_metrics方法记录训练过程中的度量指标
trainer.log_metrics("train", metrics)
# 使用trainer对象的save_metrics方法保存训练过程中的度量指标
trainer.save_metrics("train", metrics)
# 使用trainer对象的save_state方法保存训练器的当前状态，包括模型参数、优化器状态以及其他训练相关的信息
trainer.save_state()

import os
# 调用trainer对象的save_model方法，用于保存训练好的模型
trainer.save_model(os.path.join(os.path.abspath(OUTPUT_DIR),'final'))
