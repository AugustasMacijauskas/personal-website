from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


# Define an iterator over the training split of the dataset
def batch_iterator(dataset, batch_size=1000, verbose=False):
    if verbose:
        print(f"Dataset size: {len(dataset)}")

    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]


def main():
    # dataset = load_dataset("code_search_net", "python")
    # dataset = load_dataset("PleIAs/Middle-English-PD")
    # dataset = load_dataset("PleIAs/Multilingual-PD")
    # print(len(dataset["train"]))

    english_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    korean_dataset = load_dataset("lcw99/wikipedia-korean-20221001", split="train")
    code_dataset = load_dataset("code_search_net", "python", split="train", trust_remote_code=True)
    code_dataset = code_dataset.rename_column("whole_func_string", "text")  # Rename whole_func_string to text
    print(len(english_dataset), len(korean_dataset), len(code_dataset))

    final_dataset = concatenate_datasets([english_dataset, korean_dataset, code_dataset])
    final_dataset = final_dataset.shuffle(seed=42)
    print(f"{len(final_dataset)=}")
    final_dataset = final_dataset.select(range(10000))
    print(f"{len(final_dataset)=}")

    # Write dataset to file as txt line by line
    with open(Path(__file__).parent / "final_dataset.txt", "w") as f:
        for data in final_dataset:
            f.write(data["text"] + "\n")

    return

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # print(sum(len(x["text"]) for x in tqdm(final_dataset)))
    new_tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(final_dataset, verbose=True),
        len(tokenizer.get_vocab()),
        show_progress=True,
        length=len(final_dataset),
    )

    print(len(new_tokenizer.get_vocab()))
    # new_tokenizer.save_pretrained("new-llama-tokenizer-english")
    # new_tokenizer.save_pretrained(Path(__file__).parent / "new-llama-tokenizer-all")


if __name__ == "__main__":
    main()
