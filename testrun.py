import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import librosa

# 1. 加载流式数据集（这里你用的是common_voice_16_0，建议换成21版本）
train_ds = load_dataset("mozilla-foundation/common_voice_21_0", "zh-HK", split="train", streaming=True)
valid_ds = load_dataset("mozilla-foundation/common_voice_21_0", "zh-HK", split="validation", streaming=True)

# 2. 文本预处理函数（保持不变）
def prepare_dataset(batch):
    transcription = batch.get("sentence", "")
    if not transcription:
        raise ValueError(f"Empty transcription in batch: {batch}")
    if transcription.startswith('"') and transcription.endswith('"'):
        transcription = transcription[1:-1]
    if transcription and transcription[-1] not in [".", "?", "!"]:
        transcription = transcription + "."
    batch["sentence"] = transcription
    return batch

# 3. 初始化 Whisper 处理器和模型
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# 4. 应用 LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 5. 音频重采样函数（保持不变）
def resample_audio(batch, target_sr=16000):
    audio = batch["audio"]
    orig_sr = audio["sampling_rate"]
    waveform = audio["array"]
    if orig_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    batch["audio"]["array"] = waveform
    batch["audio"]["sampling_rate"] = target_sr
    return batch

# 6. 预处理函数（labels padding改为-100）
def preprocess_batch(batch):
    audio = batch["audio"]
    if "sentence" not in batch or batch["sentence"] is None or batch["sentence"].strip() == "":
        raise ValueError(f"Empty or missing sentence in batch: {batch}")

    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    labels = processor(text_target=batch["sentence"], return_tensors="pt").input_ids[0]
    return {
        "input_features": inputs.input_features[0],
        "labels": labels,
    }

# 7. 自定义 IterableDataset（保持不变）
class StreamingIterableDataset(IterableDataset):
    def __init__(self, hf_iterable):
        self.dataset = hf_iterable

    def __iter__(self):
        for example in self.dataset:
            try:
                example = prepare_dataset(example)
                if not example.get("sentence") or not example["sentence"].strip():
                    continue
                example = resample_audio(example, 16000)
                example = preprocess_batch(example)
                yield example
            except Exception as e:
                print(f"Skipping example due to error: {e}")
                continue

# 8. 自定义 collate 函数，labels padding_value改为-100
def collate_fn(batch):
    input_features = [item["input_features"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_features = torch.stack(input_features)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # -100是transformers默认ignore_index

    return {
        "input_features": input_features,
        "labels": labels,
    }

# 9. 包装数据集和 DataLoader，增加num_workers提高加载速度
train_ds = StreamingIterableDataset(train_ds)
valid_ds = StreamingIterableDataset(valid_ds)

train_loader = DataLoader(train_ds, batch_size=16, collate_fn=collate_fn, num_workers=4)
valid_loader = DataLoader(valid_ds, batch_size=16, collate_fn=collate_fn, num_workers=4)

# 10. 优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=1e-4)
num_training_steps = 3104
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# 11. 训练循环，加入混合精度支持
scaler = torch.cuda.amp.GradScaler()
gradient_accumulation_steps = 4
max_steps = num_training_steps
model.train()

global_step = 0
epoch = 0
while global_step < max_steps:
    epoch += 1
    print(f"Starting epoch {epoch}")
    train_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_loader)):
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps

        scaler.scale(loss).backward()
        train_loss += loss.item() * gradient_accumulation_steps

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()
            global_step += 1

            if global_step % 10 == 0:
                avg_loss = train_loss / max(1, ((step + 1) / gradient_accumulation_steps))
                print(f"Step {global_step} - avg training loss: {avg_loss:.4f}")

            if global_step >= max_steps:
                break

# 12. 验证循环（保持不变，加入混合精度）
model.eval()
val_loss = 0.0
val_steps = 0
with torch.no_grad():
    for batch in tqdm(valid_loader):
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        with torch.cuda.amp.autocast():
            outputs = model(input_features=input_features, labels=labels)
            val_loss += outputs.loss.item()
            val_steps += 1

avg_val_loss = val_loss / val_steps if val_steps > 0 else float("nan")
print(f"Epoch {epoch} validation loss: {avg_val_loss:.4f}")
model.train()
print("Training complete~")

# 13. 保存模型，只保存 LoRA 权重和处理器
model.save_pretrained("./lora_saver", safe_serialization=True)
processor.save_pretrained("./whisper-lora-cantonese")
