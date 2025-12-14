from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

print("load model and processor")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

print("load dummy dataset and read audio files")
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

print("generate token ids")
predicted_ids = model.generate(input_features)
print("decode token ids to text")
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
print(transcription)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)

