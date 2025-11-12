from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        message = [
            {
                "role": "system",
                "content": (
                    "You are a processional unit conversion assistant. You solve unit conversions with clear step-by-step "
                    "reasoning. Convert between units such as meters, yards, feet, inches, kilometers, miles, etc. "
                    "Always provide clear steps, and provide only the numerical result with the correct unit. "
                    "Provide final answer in the format <answer>NUMBER</answer>. Be concise."
                )
            },
            # Example 1
            {
                "role": "user",
                "content": "Convert 10 meters to feet."
            },
            {
                "role": "assistant",
                "content": (
                    "1 meter = 3.281 feet. So 10 meters * 3.281 = 32.81 feet.\n"
                    "<answer>32.81</answer>"
                )
            },
            # Example 2
            {
                "role": "user",
                "content": "A train travels 60 miles per hour for 1.5 hours. How far does it go in kilometers?"
            },
            {
                "role": "assistant",
                "content": (
                    "Step 1: Distance = speed * time = 60 mph * 1.5 h = 90 miles.\n"
                    "Step 2: 1 mile = 1.609 km. 90 * 1.609 = 144.81 km.\n"
                    "<answer>144.81</answer>"
                )
            },
            # Example 3
            {
                "role": "user",
                "content": "Convert 5 kilograms per meter to pounds per foot."
            },
            {
                "role": "assistant",
                "content": (
                    "Step 1: 1 kilogram = 2.20462 pounds. 5 kg = 11.0231 lb.\n"
                    "Step 2: 1 meter = 3.28084 feet. 1 per meter = 1 / 3.28084 per foot.\n"
                    "Step 3: 11.0231 / 3.28084 = 3.35984077249 lb/ft.\n"
                    "<answer>3.35984077249</answer>"
                )
            },
            {
                "role": "user", 
                "content": question
            }
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=False
        )

        return formatted_prompt


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
