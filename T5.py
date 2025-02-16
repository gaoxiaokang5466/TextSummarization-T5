import os
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    DataCollatorForSeq2Seq
from rouge_score import rouge_scorer  # 导入rouge_score库
import numpy as np
from transformers import pipeline
import time  # 用于计算训练时间


# 读取数据并组织成数据集格式
def load_data(txt_dir, abstract_dir):
    # 获取所有txt文件
    txt_files = sorted(os.listdir(txt_dir))
    abstract_files = sorted(os.listdir(abstract_dir))

    # 初始化空列表来存储数据
    texts = []

    for txt_file in txt_files:
        with open(os.path.join(txt_dir, txt_file), 'r', encoding='utf-8') as txt_f:
            text = txt_f.read().strip()

        texts.append(text)

    # 创建DataFrame
    data = pd.DataFrame({"content": texts})
    return Dataset.from_pandas(data)


# 加载数据集
train_data = load_data("../data/train/txt", "../data/train/abstract")
test_data = load_data("../data/test/txt", "../data/test/abstract")
dev_data = load_data("../data/dev/txt", "../data/dev/abstract")

# 合并数据集
ds = {
    "train": train_data,
    "test": test_data,
    "dev": dev_data
}
model_path = "./T5model"
# 加载英文T5分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)  # 使用英文T5模型
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


# 数据预处理
def process_func(examples):
    contents = ["summarize: " + e for e in examples["content"]]  # 修改任务提示为英文
    inputs = tokenizer(contents, max_length=384, truncation=True, padding="max_length")
    inputs["labels"] = inputs["input_ids"]  # 摘要数据作为标签
    return inputs


# 对数据集进行映射处理
tokenized_ds = {split: data.map(process_func, batched=True) for split, data in ds.items()}


# ROUGE评估指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 使用rouge_score库计算ROUGE指标
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1 = []
    rouge2 = []
    rougeL = []

    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(label, pred)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)

    return {
        "rouge-1": np.mean(rouge1),
        "rouge-2": np.mean(rouge2),
        "rouge-l": np.mean(rougeL),
    }


# 训练参数设置
training_args = Seq2SeqTrainingArguments(
    output_dir="./summary",
    evaluation_strategy="epoch",
    logging_steps=8,
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    predict_with_generate=True,
    metric_for_best_model="rouge-l",
)

# 创建Trainer对象
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["dev"],  # 使用验证集
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer)
)

# 训练模型并计算时间
start_time = time.time()  # 记录训练开始时间
train_results = trainer.train()
end_time = time.time()  # 记录训练结束时间

# 计算训练时间
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

# 打印ROUGE指标和评估损失
eval_results = trainer.evaluate()
print(f"ROUGE-1: {eval_results['eval_rouge-1']}")
print(f"ROUGE-2: {eval_results['eval_rouge-2']}")
print(f"ROUGE-L: {eval_results['eval_rouge-l']}")
print(f"Evaluation Loss: {eval_results['eval_loss']}")

# 下面部分是你不需要的，因此已注释掉：
# # 使用模型进行批量预测
# pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)

# # 批量生成摘要
# generated_summaries = []
# for idx, item in enumerate(ds["test"]):  # 加入索引
#     content = item["content"]
#     generated_summary = pipe("summarize: " + content, max_length=128, do_sample=True)[0]["generated_text"]
#     generated_summaries.append((idx, generated_summary))  # 保存原始索引

# # 根据索引排序摘要
# generated_summaries.sort(key=lambda x: x[0])  # 按照原始顺序排序

# # 打印生成的摘要
# for idx, generated_summary in generated_summaries:
#     print(f"Input {idx+1}: {ds['test'][idx]['content']}")
#     print(f"Generated Summary {idx+1}: {generated_summary}")
#     print("-" * 50)
