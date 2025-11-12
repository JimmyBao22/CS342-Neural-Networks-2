import json
from tqdm import tqdm
from .cot import CoTModel
from .data import Dataset, is_answer_valid


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    # model = CoTModel()
    model = CoTModel("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    data = Dataset("train")
    result = []

    num_return_sequences = 20

    for question, correct_answer in data:
        prompt = model.format_prompt(question)

        generated_answers = model.batched_generate(
            [prompt], 
            num_return_sequences=num_return_sequences, 
            temperature=temperature
        )[0]

        for answer in generated_answers:
            parsed_answer = model.parse_answer(answer)
            if is_answer_valid(parsed_answer, correct_answer):
                result.append([
                    question,
                    float(correct_answer),
                    answer
                ])
                break

    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
