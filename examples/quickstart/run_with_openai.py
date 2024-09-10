from openai import OpenAI


def run_with_openai(topic):
    client = OpenAI()

    result = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "user",
                "content": f"Write a haiku about '{topic}'. Don't write anything else.",
            }
        ],
    )

    print(result)


if __name__ == "__main__":
    run_with_openai("artificial intelligence")
