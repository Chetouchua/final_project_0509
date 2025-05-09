import boto3
import io
import csv
import time
import traceback
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm

# setting S3
BUCKET_NAME = "cantoneseradioso9rry"
TSV_PREFIX = "tsv_files/"  
TARGET_SR = 16000  

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.1,
    bias="none"
)

s3 = boto3.client('s3', region_name='ap-southeast-2')

def list_all_keys(bucket, prefix):
    keys = []
    continuation_token = None
    while True:
        kwargs = {'Bucket': bucket, 'Prefix': prefix, 'MaxKeys': 1000}
        if continuation_token:
            kwargs['ContinuationToken'] = continuation_token
        resp = s3.list_objects_v2(**kwargs)
        contents = resp.get('Contents', [])
        keys.extend([obj['Key'] for obj in contents])
        if resp.get('IsTruncated'):
            continuation_token = resp.get('NextContinuationToken')
        else:
            break
    return keys

def download_all_tsv_data(bucket, prefix):
    all_keys = list_all_keys(bucket, prefix)
    all_data = []
    print(f"Found {len(all_keys)} files in S3 under prefix '{prefix}'")
    for key in all_keys:
        filename = key.split('/')[-1].lower()
        if not (key.endswith('.tsv') and ("train" in filename or "test" in filename)):
            print(f"Skipping file: {key}")
            continue

        print(f"Downloading and parsing TSV file: {key}")
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            content = obj['Body'].read().decode('utf-8')
            reader = csv.DictReader(io.StringIO(content), delimiter='\t')

            if 'path' not in reader.fieldnames or 'sentence' not in reader.fieldnames:
                print(f"Warning: TSV file {key} missing required columns, skipping.")
                continue

            for row in reader:
                audio_key = row['path'].strip()
                transcription = row['sentence'].strip()
                if transcription:
                    all_data.append((audio_key, transcription))
        except Exception as e:
            print(f"Error processing {key}: {e}")

    print(f"Total samples loaded from TSV files: {len(all_data)}")
    return all_data


class WhisperDataset(Dataset):
    def __init__(self, data, processor, bucket_name, target_sr=16000):
        self.data = data
        self.processor = processor
        self.bucket_name = bucket_name
        self.target_sr = target_sr
        self.s3 = None

    def _init_s3(self):
        import boto3
        self.s3 = boto3.client('s3', region_name='ap-southeast-2')  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.s3 is None:
            self._init_s3()

        audio_key, transcription = self.data[idx]
        max_retries = 10

        for attempt in range(1, max_retries + 1):
            try:
                obj = self.s3.get_object(Bucket=self.bucket_name, Key=audio_key)
                audio_bytes = io.BytesIO(obj['Body'].read())
                waveform, sr = librosa.load(audio_bytes, sr=self.target_sr)
                break  
            except Exception as e:
                print(f"[Error] 第 {attempt} 次嘗試下載音頻失敗，Key: {audio_key}")
                print(f"錯誤訊息: {e}")
                traceback.print_exc()
                if attempt == max_retries:
                    print(f"[Error] 超過最大重試次數，跳過該音頻: {audio_key}")
                    return None
                time.sleep(2)  

        input_features = self.processor.feature_extractor(
            waveform, sampling_rate=self.target_sr, return_tensors="pt"
        ).input_features[0]

        labels = self.processor.tokenizer(
            transcription, return_tensors="pt", padding="longest", truncation=True
        ).input_ids[0]

        return {"input_features": input_features, "labels": labels}


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    input_features = torch.stack([b["input_features"] for b in batch])
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100
    )
    return {"input_features": input_features, "labels": labels}

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading processor and model...")
    processor = WhisperProcessor.from_pretrained("./whisper_lora_finetuned")
    model = WhisperForConditionalGeneration.from_pretrained("./whisper_lora_finetuned")
    model = get_peft_model(model, lora_config)
    model.to(device)

    print("Downloading and parsing all TSV files from S3...")
    all_data = download_all_tsv_data(BUCKET_NAME, TSV_PREFIX)

    dataset = WhisperDataset(all_data, processor, bucket_name=BUCKET_NAME)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    num_epochs = 3
    gradient_accumulation_steps = 4
    num_training_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    scaler = torch.amp.GradScaler()

    model.train()
    global_step = 0
    running_loss = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(tqdm(dataloader)):
            if batch is None:
                continue

            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type="cuda"):
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

    print("Saving fine-tuned model and processor...")
    model.save_pretrained("./whisper_lora_finetuned")
    processor.save_pretrained("./whisper_lora_finetuned")
    print("Training complete.")

if __name__ == "__main__":
    train()
