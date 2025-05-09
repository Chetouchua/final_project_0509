from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa


model = WhisperForConditionalGeneration.from_pretrained("/home/ruachua/venv_whisper/venv_whisper/whisper_canton/whisper_small")
processor = WhisperProcessor.from_pretrained("/home/ruachua/venv_whisper/venv_whisper/whisper_canton/whisper_small")

if hasattr(model.config, "forced_decoder_ids"):
    del model.config.forced_decoder_ids

audio, sr = librosa.load("target_audio.wav", sr=16000)

inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

generated_ids = model.generate(
    inputs.input_features,
    task="translate",    
    language="zh",       
    max_length=2056,      
    num_beams=15          
)

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("翻译结果：", transcription)
