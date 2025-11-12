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
                    "You are a helpful assistant that solves math word problems. You solve unit conversions with clear step-by-step "
                    "reasoning. Always provide clear steps, and provide only the numerical result with the correct unit. "
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
                    "One meter is equal to 3.281 feet, so 10 meters corresponds to 10 * 3.281 = 32.81 feet.\n"
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
                    "The train travels a distance equal to speed times time, which is 60 mph * 1.5 hours = 90 miles.\n"
                    "Since 1 mile equals 1.609 kilometers, the distance in kilometers is 90 * 1.609 = 144.81 km.\n"
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
                    "One kilogram is equal to 2.20462 pounds, so 5 kilograms corresponds to 5 * 2.20462 = 11.0231 pounds per meter. \n"
                    "Since 1 meter equals 3.28084 feet, this converts to 11.0231 / 3.28084 = 3.35984077249 pounds per foot.\n"
                    "<answer>3.35984077249</answer>"
                )
            },
            # Example 4
            {
                "role": "user",
                "content": "Convert 3 mi/h to its equivalent in in/s."
            },
            {
                "role": "assistant",
                "content": (
                    "One mile is 5280 feet, and one foot is 12 inches, so one mile equals 5280 * 12 = 63,360 inches.\n"
                    "A speed of 3 mi/h corresponds to 3 * 63,360 = 190,080 inches per hour. \n"
                    "Dividing by 3600 seconds per hour gives 190,080 / 3600 = 52.8 inches per second.\n"
                    "<answer>52.8</answer>"
                )
            },
            # Example 5
            {
                "role": "user",
                "content": "Convert 3 mph to its equivalent in km/h."
            },
            {
                "role": "assistant",
                "content": (
                    "Since 1 mile is equal to 1.60934 kilometers, 3 miles corresponds to 3 * 1.60934 = 4.828031999999999 kilometers.\n"
                    "Because the speed is per hour, the conversion directly gives 3 mph = 4.828031999999999 km/h.\n"
                    "<answer>4.828031999999999</answer>"
                )
            },
            # Example 6
            {
                "role": "user",
                "content": "Convert 2 hours to its equivalent in minutes."
            },
            {
                "role": "assistant",
                "content": (
                    "Since 1 hour is equal to 60 minutes, 2 hours correspond to 2 * 60 = 120 minutes.\n"
                    "<answer>120.0</answer>"
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
