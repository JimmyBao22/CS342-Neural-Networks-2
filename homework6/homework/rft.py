from .base_llm import BaseLLM
from .sft import test_model, TokenizedDataset, format_example
from .data import Dataset, benchmark
import torch


class RFTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        RFT models are trained on raw questions without chat templates.
        Return the question as-is.
        """
        return question


def load() -> RFTModel:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = RFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str = "./homework/rft_model",
    **kwargs,
):
    # Reuse much of the SFT code here
    # raise NotImplementedError()
    llm = RFTModel()

    from peft import get_peft_model, LoraConfig
    from transformers import TrainingArguments, Trainer
    
    lora_config = LoraConfig(
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32
    )

    llm.model = get_peft_model(llm.model, lora_config)
    if torch.cuda.is_available():
        llm.model.enable_input_require_grads()
    
    train_data = Dataset("train")
    tokenized = TokenizedDataset(llm.tokenizer, train_data, format_example)

    training_args = TrainingArguments(
        gradient_checkpointing=True,
        learning_rate=1e-3,
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=32
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=llm.tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)

    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
