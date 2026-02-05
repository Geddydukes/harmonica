from datasets import load_dataset
ds = load_dataset("confit/vctk-full", split="train", streaming=True)
it = iter(ds)
sample = next(it)
print(sample.keys())
print(sample["text"][:80])
if "audio" in sample:
    print(sample["audio"]["sampling_rate"], len(sample["audio"]["array"]))
else:
    print("(no 'audio' in streaming sample; keys:", list(sample.keys()), ")")
