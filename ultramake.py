import boto3
import io
import csv
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
import evaluate 

# setting your s3
BUCKET_NAME = "cantoneseradioso9rry"
TSV_KEY = "tsv_files/train.tsv"
REGION = "ap-east-1"

s3 = boto3.client('s3', region_name=REGION)

def download_tsv(tsv_key, start=0, batch_size=10):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=tsv_key)
    content = obj['Body'].read().decode('utf-8')
    reader = csv.DictReader(io.StringIO(content), delimiter='\t')
    data = []
    for i, row in enumerate(reader):
        if i < start:
            continue
        if i >= start + batch_size:
            break
        audio_path = row['path'].strip()
        transcription = row['sentence'].strip()
        if transcription:
            data.append((audio_path, transcription))
    return data

class WhisperDataset(Dataset):
    def __init__(self, data, processor, target_sr=16000):
        self.data = data
        self.processor = processor
        self.target_sr = target_sr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, transcription = self.data[idx]
        waveform, sr = self.load_audio_from_s3(audio_path)
        if waveform is None:
            return None
        if sr != self.target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sr)

        input_features = self.processor.feature_extractor(waveform, sampling_rate=self.target_sr, return_tensors="pt").input_features[0]
        labels = self.processor.tokenizer(transcription, return_tensors="pt", padding="longest", truncation=True).input_ids[0]

        return {"input_features": input_features, "labels": labels}

    def load_audio_from_s3(self, audio_key, max_retry=3):
        for attempt in range(max_retry):
            try:
                obj = s3.get_object(Bucket=BUCKET_NAME, Key=audio_key)
                audio_bytes = io.BytesIO(obj['Body'].read())
                waveform, sr = sf.read(audio_bytes)
                return waveform, sr
            except Exception as e:
                print(f"Attempt {attempt+1} failed to load audio {audio_key}: {e}")
        return None, None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    input_features = torch.stack([b['input_features'] for b in batch])
    labels = torch.nn.utils.rnn.pad_sequence([b['labels'] for b in batch], batch_first=True, padding_value=-100)
    return {"input_features": input_features, "labels": labels}

def evaluate_cer(model, processor, dataset, device):
    cer_metric = evaluate.load("cer")
    model.eval()
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating CER"):
            if batch is None:
                continue
            input_features = batch["input_features"].to(device)
            labels = batch["labels"]

            generated_ids = model.generate(input_features)
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            label_strs = processor.batch_decode(labels, skip_special_tokens=True)

            cer_metric.add_batch(predictions=preds, references=label_strs)

    cer_score = cer_metric.compute()
    print(f"CER: {cer_score:.4f}")
    return cer_score

def train():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

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

    obj = s3.get_object(Bucket=BUCKET_NAME, Key=TSV_KEY)
    content = obj['Body'].read().decode('utf-8')
    total_data = len(list(csv.DictReader(io.StringIO(content), delimiter='\t')))
    print(f"Total dataset size: {total_data}")

    batch_size_data = 100 #100 radios per round
    num_epochs = 3
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler()
    gradient_accumulation_steps = 4

    global_step = 0
    running_loss = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        start_idx = 0
        while start_idx < total_data:
            data = download_tsv(TSV_KEY, start=start_idx, batch_size=batch_size_data)
            if len(data) == 0:
                break

            dataset = WhisperDataset(data, processor)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4)

            num_training_steps = len(dataloader) * num_epochs
            lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

            model.train()
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(dataloader, desc=f"Training batch starting at {start_idx}")):
                if batch is None:
                    continue
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast(device_type=device):
                    outputs = model(input_features=input_features, labels=labels)
                    loss = outputs.loss / gradient_accumulation_steps

                scaler.scale(loss).backward()
                running_loss += loss.item() * gradient_accumulation_steps

                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    global_step += 1

                    if global_step % 10 == 0:
                        avg_loss = running_loss / 10
                        print(f"Step {global_step} - avg loss: {avg_loss:.4f}")
                        running_loss = 0.0

            start_idx += batch_size_data

    model.save_pretrained("./whisper_lora_finetuned", safe_serialization=True)
    processor.save_pretrained("./whisper_lora_finetuned")

    print("Evaluating CER on validation subset...")
    eval_data = download_tsv(TSV_KEY, start=0, batch_size=1000) 
    eval_dataset = WhisperDataset(eval_data, processor)
    evaluate_cer(model, processor, eval_dataset, device)

if __name__ == "__main__":
    train()
