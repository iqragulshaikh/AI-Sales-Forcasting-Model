import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from retrieval import retrieve_context

# Load API key from .env file
load_dotenv()

# The grounding system prompt - tells the AI exactly how to behave
SYSTEM_PROMPT = """You are a restaurant sales analytics assistant.

Rules:
1. Use ONLY the forecast table provided below as the source for any numeric sales values.
2. Use the retrieved project documents only to explain definitions, assumptions, and limitations - never for numbers.
3. Never invent sales figures, dates, branch IDs, metrics, or confidence intervals that are not present in the forecast table or documents.
4. If the user asks something not covered by the forecast table or documents, clearly say the information is not available.
5. Keep answers concise and always mention the forecast horizon.
"""


def explain_forecast(forecast_df, metrics, question):
    """
    Takes the numeric forecast table, evaluation metrics, and a user question.
    Retrieves relevant project documents and asks Gemini to generate a
    grounded explanation. Falls back to a plain message if the API is unavailable.
    """
    # Check if the API key exists - if not, use the fallback immediately
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return _fallback_response(forecast_df)

    try:
        # Step 1: Retrieve relevant document chunks for this question
        retrieved_chunks = retrieve_context(question, k=3)
        context_text = "\n\n".join([chunk["content"] for chunk in retrieved_chunks])

        # Step 2: Convert the forecast table to plain text so the AI can read it
        forecast_text = forecast_df.to_string(index=False)

        # Step 3: Build the full prompt combining system rules, forecast data,
        # retrieved documents, and the user's question
        full_prompt = f"""{SYSTEM_PROMPT}

Forecast Table:
{forecast_text}

Evaluation Metrics:
{metrics}

Retrieved Project Documents:
{context_text}

User Question: {question}

Answer:"""

        # Step 4: Call the Gemini model
        model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
        response = model.invoke(full_prompt)

        # Newer Gemini models return content as a list of blocks instead of plain text
        if isinstance(response.content, str):
            return response.content
        else:
            text_parts = [
                block.get("text", "") 
                for block in response.content 
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "".join(text_parts)

    except Exception as e:
        # If anything goes wrong with the API call, fall back gracefully
        print(f"AI explanation unavailable due to error: {e}")
        return _fallback_response(forecast_df)


def _fallback_response(forecast_df):
    """
    Returns the numeric forecast without an AI explanation.
    Used when the API key is missing or the API call fails.
    """
    forecast_text = forecast_df.to_string(index=False)
    return (
        "AI explanation is currently unavailable (missing or invalid API key). "
        "Here is the numeric forecast instead:\n\n" + forecast_text
    )


# Quick manual test - only runs if you execute this file directly
if __name__ == "__main__":
    import pandas as pd

    # Sample forecast table to test with (mimics what Iqra's model will produce)
    sample_forecast = pd.DataFrame({
        "date": ["2025-12-25", "2025-12-26", "2025-12-27"],
        "branch_id": ["B001", "B001", "B001"],
        "predicted_sales": [113259.63, 100815.19, 142128.53]
    })

    sample_metrics = "MAE: 45.2, RMSE: 60.1"

    test_question = 'What is the forecast for branch B999?'
    answer = explain_forecast(sample_forecast, sample_metrics, test_question)

    print("\n--- AI Explanation ---")
    print(answer)