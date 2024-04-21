from datasets import load_dataset
from transformers import AutoTokenizer


# Define an iterator over the training split of the dataset
def batch_iterator(dataset, batch_size=1000, verbose=False):
    if verbose:
        print(f"Dataset size: {len(dataset)}")

    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]


def main():
    # dataset = load_dataset("PleIAs/Multilingual-PD")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    final_dataset = dataset["train"]
    print(f"{len(final_dataset)=}")
    # print(sum(len(x["text"].split()) for x in final_dataset))
    new_tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator(final_dataset, verbose=True),
        len(tokenizer.get_vocab()),
        show_progress=True,
        length=len(final_dataset),
    )

    print(len(new_tokenizer.get_vocab()))
    new_tokenizer.save_pretrained("new-llama-tokenizer-v1")


if __name__ == "__main__":
    main()
