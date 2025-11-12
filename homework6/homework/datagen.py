import json
from tqdm import tqdm
from .cot import CoTModel
from .data import Dataset, is_answer_valid


def generate_dataset(output_json: str = "data/rft.json", oversample: int = 10, temperature: float = 0.6):
    # model = CoTModel()
    model = CoTModel("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    data = Dataset("train")
    result = []
    iter = 0

    # max_prompts_per_batch = 16
    # questions = [q for q, _ in data]
    # answers = [a for _, a in data]

    # for i in tqdm(range(0, len(questions), max_prompts_per_batch), desc="Generating Samples"):
    #     batch_questions = questions[i : i + max_prompts_per_batch]
    #     batch_answers = answers[i : i + max_prompts_per_batch]

    #     # Generate multiple diverse completions
    #     batch_generations = model.batched_generate(
    #         batch_questions,
    #         num_return_sequences=oversample,
    #         temperature=temperature
    #     )

    #     # Evaluate correctness and keep only valid generations
    #     for question, correct_answer, generated_answer in zip(batch_questions, batch_answers, batch_generations):
    #         for answer in generated_answer:
    #             parsed_answer = model.parse_answer(answer)
    #             if is_answer_valid(parsed_answer, correct_answer):
    #                 result.append([
    #                     question,
    #                     float(correct_answer),
    #                     answer.strip()
    #                 ])
    #                 break

    for i in tqdm(range(0, len(data)), desc="Generating Samples"):
        question = data[i][0]
        correct_answer = data[i][1]

        prompt = model.format_prompt(question)

        generated_answers = model.batched_generate(
            [prompt], 
            num_return_sequences=oversample, 
            temperature=temperature
        )[0]

        for answer in generated_answers:
            parsed_answer = model.parse_answer(answer)
            if is_answer_valid(parsed_answer, correct_answer):
                result.append([
                    question,
                    float(correct_answer),
                    answer.strip()
                ])
                break

    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    print("Saved files from datagen")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
