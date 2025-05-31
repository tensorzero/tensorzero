from openai import OpenAI
import pytest

@pytest.mark.asyncio
async def test_openai_pdf():
    client = OpenAI(base_url="http://localhost:3000/v1", api_key="dummy")
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Analyze this PDF.", "pdf": {"file_path": "tests/fixtures/sample.pdf"}}]
    )
    assert response.choices[0].message.content
