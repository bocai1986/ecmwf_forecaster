import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(
        base_url="http://10.0.74.56:11434/v1",
        api_key="ollama",
    )

    response = await client.chat.completions.create(
        model="gpt-oss:latest",
        # model = "qwen3:8b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你好！请介绍一下你自己"},
        ],
    )

    print("✅ 调用成功：")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(main())






