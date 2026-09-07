# Prompt Design — Rakhi (LangChain, Vector DB, Prompt Engineering)

## System Prompt (Draft v1)

You are a restaurant sales analytics assistant.

Rules:
1. Use ONLY the forecast table (columns: date, branch_id, predicted_sales) 
   as the source for any numeric sales values you mention.
2. Use the retrieved project documents only to explain definitions, 
   assumptions, and limitations — never for numbers.
3. Never invent sales figures, dates, branch IDs, metrics, or confidence 
   intervals that are not present in the forecast table or documents.
4. If the user asks something not covered by the forecast table or 
   documents, clearly say the information is not available — do not guess.
5. Keep answers concise and always mention the forecast horizon 
   (e.g. "based on the 7-day forecast...").

## Reference Documents Status

Data dictionary: available (extracted from Zara's notebook and added 
to docs/data_dictionary.md).

Technical write-up: not yet available from the team. Working with 
placeholder assumptions until Muntaha pushes this file.

## System Architecture

- `src/retrieval.py` — loads project documents (.md files) from the docs/ 
  folder, splits them into chunks, creates embeddings using Gemini's 
  `gemini-embedding-001` model, and stores them in a FAISS vector store. 
  Includes `build_knowledge_base()` and `retrieve_context()` functions.
- `src/llm_explainer.py` — combines the forecast table, evaluation metrics, 
  and retrieved document context into a single grounded prompt, then calls 
  Gemini's `gemini-3.6-flash` chat model via `explain_forecast()`. Falls 
  back to a plain numeric forecast message if the API key is missing or 
  the API call fails.

## Tested Examples

**Test 1 — Numeric question (grounded in forecast table)**
Q: "What is the expected sales for branch B001 on 2026-09-06?"
A: "The expected sales for branch B001 on 2026-09-06 is 1,200.50. 
(Forecast horizon: 2026-09-06 to 2026-09-08)"

**Test 2 — Highest sales day (grounded in forecast table)**
Q: "Which day has the highest expected sales?"
A: "Within the forecast horizon of 2026-09-06 to 2026-09-08, the day with 
the highest expected sales for branch B001 is 2026-09-07, with predicted 
sales of 1,350.75."

**Test 3 — Assumptions question (grounded in retrieved documents)**
Q: "What assumptions does this model make?"
A: "Based on the project documents, the model relies on the following key 
assumptions: Data Adaptation — it assumes an adapted retail sales dataset 
can serve as a prototype for restaurant performance, rather than using 
actual restaurant transaction data. Synthetic Features — it assumes 
synthetic/documented features for temperature values and holiday 
indicators, as real source fields were unavailable. Artificial Branch 
IDs — it assumes generated branch IDs (such as B001) to structure the 
forecasting prototype. (Forecast horizon: 2026-09-06 to 2026-09-08)"

**Test 4 — Unavailable branch (correctly refuses to guess)**
Q: "What is the forecast for branch B999?"
A: "The forecast information for branch B999 is not available. The 
forecast table only includes data for branch B001 for the forecast 
horizon of 2026-09-06 to 2026-09-08."

**Test 5 — API fallback (missing API key)**
When GOOGLE_API_KEY is unavailable, the system returns the numeric 
forecast without crashing:
"AI explanation is currently unavailable (missing or invalid API key). 
Here is the numeric forecast instead: [forecast table shown]"

## Notes / Known Limitations

- Currently tested only with a sample/dummy forecast table (Iqra's real 
  forecast output not yet integrated as of Day 3).
- Technical write-up not yet available — some assumptions may need 
  updating once it is pushed.