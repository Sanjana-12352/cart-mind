import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage


async def normalize_query(query: str) -> str:

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=200
    )

    system_content = (
        "You are a query normalization expert for an e-commerce search engine. "
        "Fix spelling mistakes and typos in user search queries. "
        "Keep the meaning exactly the same. "
        "Return ONLY the cleaned query as plain text. "
        "No explanation, no punctuation changes, just the clean query. "
        "Examples: "
        "'dinning tabel setup' → 'dining table setup' "
        "'bithday prty supplis' → 'birthday party supplies' "
        "'ofice holidy prty' → 'office holiday party' "
        "'studi deks fr studnt' → 'study desk for student' "
        "'pls i need sum pens' → 'pens' "
        "'pen' → 'pen' "
        "'chair' → 'chair' "
    )

    human_content = f"Clean this query: {query}"

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content)
    ]

    response = await llm.ainvoke(messages)
    normalized = response.content.strip()

    print(f" Normalized: '{query}' → '{normalized}'")
    return normalized