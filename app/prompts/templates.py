"""
All prompt templates for the RAG chatbot system.
Designed to minimize hallucination and maximize factual grounding.
"""

SYSTEM_PROMPT = """You are an expert AI research assistant with access to a curated knowledge base and real-time web search.

CORE PRINCIPLES:
- Answer ONLY from the provided context. Do not hallucinate facts not present in the sources.
- If the context is insufficient, clearly state: "Based on available information, I cannot fully answer this."
- Always cite your sources using [1], [2], ... notation matching the provided sources.
- Be concise, precise, and structured. Use markdown for clarity.
- Reason step-by-step for complex questions before giving your final answer.

CITATION RULES:
- Every factual claim must be backed by a citation [N].
- Place citations inline, immediately after the claim they support.
- Do not fabricate or modify source content.

TONE: Professional, clear, and helpful. Avoid filler phrases like "Great question!" or "Certainly!"
"""

QUERY_REWRITE_PROMPT = """You are a query optimization expert. Rewrite the user's question to be more precise and retrievable.

CONVERSATION HISTORY:
{history}

ORIGINAL QUERY: {query}

INSTRUCTIONS:
1. Resolve ambiguous pronouns and references using conversation history.
2. Add specificity — expand abbreviations and clarify the subject, but do NOT add or change any dates, years, or time qualifiers.
3. Preserve the user's intent exactly.
4. Output ONLY the rewritten query as a single line. No explanation, no preamble, no labeling.

REWRITTEN QUERY:"""

MULTI_QUERY_PROMPT = """Generate {n} diverse search queries to retrieve comprehensive information for answering this question.

QUESTION: {query}

INSTRUCTIONS:
- Each query should approach the question from a different angle.
- Use different keywords, synonyms, and phrasings.
- Keep each query concise (under 15 words).
- Output ONLY the queries, one per line. No numbering, no explanation.

QUERIES:"""

ROUTER_PROMPT = """Determine the best retrieval strategy for this query.

QUERY: {query}

OPTIONS:
- "vector_only"  — question is about uploaded documents (technical docs, reports, PDFs)
- "web_only"     — question requires current/real-time information (news, prices, events)
- "both"         — question benefits from both document knowledge and web context
- "direct"       — simple factual/conversational question that needs no retrieval

CONVERSATION HISTORY: {history}

Respond with EXACTLY one of: vector_only, web_only, both, direct
No explanation. No punctuation.

ROUTE:"""

RAG_ANSWER_PROMPT = """You are answering a question using the retrieved context below.
Context may come from uploaded documents, web search results, or both.
Read ALL excerpts carefully before answering — the answer may be spread across multiple chunks.

RETRIEVED CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

QUESTION: {question}

INSTRUCTIONS:
- Base your answer ENTIRELY on the retrieved context above.
- If context includes both document [Document] and web [Web] sources:
  * Prefer information from Document sources for facts about uploaded content.
  * Use Web sources to fill gaps or add broader context the document doesn't cover.
- If the answer is directly stated in any chunk, quote or paraphrase it precisely.
- Synthesize across multiple chunks if the answer is spread out.
- Cite every fact with [N] matching the source number in the context.
- If the context genuinely does not contain the answer, say clearly: "The available sources do not contain information about this."
- Use markdown for structure. Be thorough — do not omit relevant details found in the context.

ANSWER:"""

VALIDATION_PROMPT = """Evaluate whether this answer is factually grounded and complete.

QUESTION: {question}

CONTEXT PROVIDED:
{context}

ANSWER GIVEN:
{answer}

EVALUATE:
1. Is every factual claim in the answer supported by the context? (yes/no)
2. Are there any hallucinations or unsupported claims? (yes/no)
3. Does the answer address the question completely? (yes/no)

Respond with JSON only:
{{"is_valid": true/false, "reason": "brief explanation", "missing": "what's missing if invalid"}}

JSON:"""

CITATION_EXTRACTION_PROMPT = """Extract citation references from this answer and map them to sources.

ANSWER: {answer}

SOURCES:
{sources}

Return a JSON list of used citation indices (1-based): [1, 3, 5]
Only list indices that are actually cited in the answer.

JSON:"""

WEB_SEARCH_QUERY_PROMPT = """Optimize this query specifically for web search to find current, authoritative information.

ORIGINAL QUERY: {query}

OUTPUT: A single optimized web search query. No explanation.

SEARCH QUERY:"""
