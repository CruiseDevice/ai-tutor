"""Adaptive prompt construction and query classification.

Extracted from ChatService. This module owns:
- classifying a user query (type + complexity + CoT flag) via a helper LLM call,
- building the adaptive system prompt (base prompt, chain-of-thought section,
  few-shot examples, annotation/citation instructions), and
- exposing the trimmed template used purely for token-budget estimation.

The classifier reads app settings (feature flags, helper model) directly; the
prompt builders are pure functions of their arguments. A process-wide singleton
is exposed via `get_prompt_builder()`, mirroring `get_annotation_service()`.
"""
import json
import logging
from typing import Dict

from ..config import settings
from .llm import Provider, get_llm_client, pick_helper_model
from .retry_utils import async_retry_openai_call

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Classify queries and build adaptive tutor system prompts."""

    async def classify_query_type(
        self,
        query: str,
        user_api_key: str,
        provider: Provider | None = None,
    ) -> Dict:
        """
        Classify query type and complexity to enable adaptive prompting.

        Returns dict with:
        - query_type: "factual", "analytical", "comparative", "follow-up", or "clarification"
        - complexity: "simple", "moderate", "complex"
        - requires_cot: bool (whether chain-of-thought prompting should be used)
        """
        # Skip if query classification is disabled
        if not settings.ENABLE_QUERY_CLASSIFICATION:
            return {
                "query_type": "factual",
                "complexity": "simple",
                "requires_cot": False
            }

        try:
            prov = provider or Provider.OPENAI
            client = get_llm_client(prov, user_api_key)
            model = pick_helper_model(prov)

            classification_prompt = """Analyze this query and classify it.

Query: "{query}"

Respond with ONLY a JSON object in this exact format:
{{
  "query_type": "<one of: factual, analytical, comparative, follow-up, clarification>",
  "complexity": "<one of: simple, moderate, complex>",
  "requires_cot": <true or false>
}}

Query types:
- factual: Asking for specific facts, definitions, or information
- analytical: Requiring analysis, interpretation, or synthesis
- comparative: Comparing concepts, ideas, or items
- follow-up: Building on previous conversation context
- clarification: Seeking clarification on previous responses

Complexity levels:
- simple: Single concept, straightforward answer
- moderate: Multiple concepts, some reasoning required
- complex: Deep analysis, multiple perspectives, synthesis needed

Requires chain-of-thought (COT):
- true: For moderate/complex queries requiring step-by-step reasoning
- false: For simple queries with straightforward answers""".format(query=query)

            async def _create_completion():
                return await client.complete(
                    system_prompt="You are a query classification assistant that outputs only valid JSON.",
                    messages=[{"role": "user", "content": classification_prompt}],
                    model=model,
                    temperature=0.3,  # Low temperature for consistent classification
                    max_tokens=100,
                )

            response_text = await async_retry_openai_call(
                _create_completion,
                max_attempts=3,
                initial_wait=1.0,
                max_wait=30.0
            )

            response_text = response_text.strip()

            # Parse JSON response
            result = json.loads(response_text)

            # Validate and set defaults
            valid_types = ["factual", "analytical", "comparative", "follow-up", "clarification"]
            valid_complexity = ["simple", "moderate", "complex"]

            query_type = result.get("query_type", "factual")
            if query_type not in valid_types:
                query_type = "factual"

            complexity = result.get("complexity", "simple")
            if complexity not in valid_complexity:
                complexity = "simple"

            requires_cot = result.get("requires_cot", False)

            # Override requires_cot based on settings and complexity threshold
            if settings.ENABLE_CHAIN_OF_THOUGHT:
                complexity_levels = {"simple": 0, "moderate": 1, "complex": 2}
                threshold_level = complexity_levels.get(settings.COT_COMPLEXITY_THRESHOLD, 1)
                current_level = complexity_levels.get(complexity, 0)
                requires_cot = current_level >= threshold_level
            else:
                requires_cot = False

            classification = {
                "query_type": query_type,
                "complexity": complexity,
                "requires_cot": requires_cot
            }

            logger.info(f"Query classification: {classification}")
            return classification

        except Exception as e:
            logger.warning(f"Failed to classify query: {e}. Using defaults.")
            return {
                "query_type": "factual",
                "complexity": "simple",
                "requires_cot": False
            }

    def build_system_prompt(
        self,
        context_text: str,
        query_type: str,
        complexity: str,
        requires_cot: bool,
    ) -> str:
        """
        Build adaptive system prompt with few-shot examples and chain-of-thought instructions.

        Args:
            context_text: Formatted document chunks with page numbers
            query_type: Classified query type (factual, analytical, comparative, etc.)
            complexity: Query complexity level (simple, moderate, complex)
            requires_cot: Whether to include chain-of-thought prompting

        Returns:
            Complete system prompt tailored to the query
        """
        # Base system prompt
        prompt = f"""You are an AI tutor helping a student understand a PDF document.
You have access to the following document chunks that are relevant to the student's question:

{context_text}

When referring to content, always cite the page number like [Page X].
Make sure to use the correct page number for each piece of information.

IMPORTANT: You should actively help students with code implementation, examples, and practical applications based on the document content.
When students ask for code implementations (in any programming language), algorithms, or technical examples:
- Provide clear, well-commented code examples based on concepts from the document
- Explain how the code relates to the concepts discussed in the document
- Cite the relevant pages that describe the underlying concepts
- Use code blocks with appropriate language tags (e.g., ```c, ```python, ```javascript)
- If the document describes algorithms or mathematical concepts, translate them into working code when requested
"""

        # Add chain-of-thought section for complex queries
        if requires_cot:
            prompt += self._get_chain_of_thought_section(query_type)

        # Add formatting instructions
        prompt += """
IMPORTANT FORMATTING INSTRUCTIONS:
1. Use markdown to highlight important concepts, terms, or phrases by making them **bold** or using *italics*.
2. For direct quotes from the document, use > blockquote formatting.
3. When referring to specific sections, use [Page X] to cite the page number.
4. Use bullet points or numbered lists for step-by-step explanations.
5. For critical information or warnings, use "⚠️" at the beginning of the paragraph.
6. For code examples, use code blocks with language tags (e.g., ```c, ```python, ```javascript) and include comments explaining how concepts from the document are implemented.

"""

        # Add few-shot examples (helps with annotation quality)
        prompt += self._get_few_shot_examples()

        # Add annotation instructions
        prompt += """

PDF ANNOTATION FEATURE - CRITICAL INSTRUCTIONS:
You MUST identify specific parts of the document that are relevant to your answer.
At the END of your response, ALWAYS include an ANNOTATIONS section with the following JSON format:

```annotations
[
  {
    "pageNumber": <page number from context above>,
    "type": "highlight",
    "textToHighlight": "<3-10 word phrase copied exactly from the document>",
    "explanation": "<why this text answers the question>"
  }
]
```

IMAGE ANNOTATION FORMAT:
```annotations
[
  {
    "pageNumber": <page number from context above>,
    "type": "circle",
    "imageChunkId": "<Image ID from context>",
    "bbox": [x0, y0, x1, y1],
    "explanation": "<why this image answers the question>"
  }
]
```

ANNOTATION RULES - FOLLOW STRICTLY:
1. ALWAYS include at least 1 annotation when you reference document content
2. The "pageNumber" MUST match a page number from the [Page X] citations above
3. For TEXT: "textToHighlight" MUST be a short phrase (3-10 words) that appears EXACTLY in the document chunks above
4. For IMAGES: Copy the bbox coordinates and imageChunkId exactly from the context above
5. Use type "highlight" for text (most common), "circle" for images/diagrams, "box" for tables
6. Copy the exact words from the document - do not paraphrase or modify them
7. Include 1-3 annotations per response, focusing on the most important points
8. Each annotation's "explanation" should clearly connect the highlighted text to the user's question

CITATION VERIFICATION:
- Double-check that all [Page X] citations in your response match the page numbers in the context above
- Ensure every annotation's pageNumber corresponds to a chunk you actually used
- If you cite information, include a corresponding annotation for that text

Make your responses helpful, clear, and educational. When students ask for code implementations or technical examples,
provide them based on the concepts described in the document. Always ground your code examples in the theoretical
concepts from the document and cite the relevant pages.

If the context doesn't contain enough information to answer the question, say you don't have enough information
from the document and suggest looking at other pages. However, if the document contains relevant concepts that
can be applied to answer the question (even if not explicitly shown as code), you should help translate those
concepts into practical implementations when requested.
"""

        return prompt

    def _get_few_shot_examples(self) -> str:
        """
        Get few-shot examples demonstrating correct annotation format.

        These examples show the LLM how to:
        - Match text exactly from source material
        - Format annotations correctly with page numbers
        - Use different annotation types appropriately
        - Link annotations to the question being answered
        """
        return """
# Few-Shot Examples

## Example 1: Factual Query with Highlight Annotation

**User Question**: "What is photosynthesis?"

**Context**:
[Page 12]: Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. Photosynthesis in plants generally involves the green pigment chlorophyll and generates oxygen as a byproduct.

**Assistant Response**:
Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water [Page 12]. This process generally involves the green pigment chlorophyll and generates oxygen as a byproduct [Page 12].

**Annotations**:
```json
{
  "annotations": [
    {
      "type": "highlight",
      "text": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water.",
      "pageNumber": 12,
      "explanation": "This sentence provides the core definition of photosynthesis that directly answers the user's question."
    },
    {
      "type": "highlight",
      "text": "Photosynthesis in plants generally involves the green pigment chlorophyll and generates oxygen as a byproduct.",
      "pageNumber": 12,
      "explanation": "This sentence explains the key components and byproducts of photosynthesis, providing additional context to the definition."
    }
  ]
}
```

## Example 2: Analytical Query with Multiple Annotations

**User Question**: "How does the mitochondria produce energy?"

**Context**:
[Page 45]: The mitochondria are known as the powerhouse of the cell. Through a process called cellular respiration, mitochondria break down glucose molecules to produce ATP (adenosine triphosphate), which serves as the cell's primary energy currency.

[Page 46]: The process occurs in three main stages: glycolysis, the Krebs cycle, and the electron transport chain. The electron transport chain is the most efficient stage, producing approximately 34 ATP molecules per glucose molecule.

**Assistant Response**:
The mitochondria produce energy through cellular respiration, where they break down glucose molecules to produce ATP (adenosine triphosphate), the cell's primary energy currency [Page 45]. This process occurs in three main stages: glycolysis, the Krebs cycle, and the electron transport chain [Page 46]. The electron transport chain is the most efficient stage, producing approximately 34 ATP molecules per glucose molecule [Page 46].

**Annotations**:
```json
{
  "annotations": [
    {
      "type": "box",
      "text": "Through a process called cellular respiration, mitochondria break down glucose molecules to produce ATP (adenosine triphosphate), which serves as the cell's primary energy currency.",
      "pageNumber": 45,
      "explanation": "This describes the fundamental mechanism by which mitochondria produce energy, introducing the key term ATP."
    },
    {
      "type": "highlight",
      "text": "The process occurs in three main stages: glycolysis, the Krebs cycle, and the electron transport chain.",
      "pageNumber": 46,
      "explanation": "This breaks down the energy production process into its three main stages, providing a systematic answer."
    },
    {
      "type": "circle",
      "text": "The electron transport chain is the most efficient stage, producing approximately 34 ATP molecules per glucose molecule.",
      "pageNumber": 46,
      "explanation": "This highlights the most efficient stage and quantifies the energy output, showing the scale of ATP production."
    }
  ]
}
```

## Example 3: Comparative Query

**User Question**: "What's the difference between mitosis and meiosis?"

**Context**:
[Page 78]: Mitosis is a type of cell division that results in two daughter cells, each having the same number of chromosomes as the parent nucleus. It is used for growth and repair in multicellular organisms.

[Page 79]: Meiosis, in contrast, is a type of cell division that reduces the chromosome number by half, creating four daughter cells. This process is essential for sexual reproduction and genetic diversity.

**Assistant Response**:
Mitosis and meiosis are both types of cell division, but they differ significantly in their outcomes and purposes. Mitosis results in two daughter cells with the same number of chromosomes as the parent cell and is used for growth and repair [Page 78]. In contrast, meiosis reduces the chromosome number by half, creating four daughter cells, and is essential for sexual reproduction and genetic diversity [Page 79].

**Annotations**:
```json
{
  "annotations": [
    {
      "type": "highlight",
      "text": "Mitosis is a type of cell division that results in two daughter cells, each having the same number of chromosomes as the parent nucleus. It is used for growth and repair in multicellular organisms.",
      "pageNumber": 78,
      "explanation": "This defines mitosis and explains its purpose, establishing one side of the comparison."
    },
    {
      "type": "highlight",
      "text": "Meiosis, in contrast, is a type of cell division that reduces the chromosome number by half, creating four daughter cells. This process is essential for sexual reproduction and genetic diversity.",
      "pageNumber": 79,
      "explanation": "This defines meiosis and contrasts it with mitosis, showing the key differences in chromosome number and purpose."
    }
  ]
}
```

---

**Key Principles Demonstrated**:
1. **Exact Text Matching**: Always copy text exactly as it appears in the source material
2. **Page Number Accuracy**: Ensure page numbers in annotations match the source chunks
3. **Annotation Types**: Use 'highlight' for key information, 'box' for processes/mechanisms, 'circle' for important data/numbers
4. **Clear Explanations**: Each annotation should explain WHY the text is relevant to answering the question
5. **Comprehensive Coverage**: Include all major points from the answer in the annotations
"""

    def _get_chain_of_thought_section(self, query_type: str) -> str:
        """
        Get chain-of-thought prompting instructions for complex queries.

        Args:
            query_type: The classified type of query (factual, analytical, comparative, etc.)

        Returns:
            Chain-of-thought prompt section tailored to the query type
        """
        base_cot = """
# Chain-of-Thought Reasoning

For this query, use systematic step-by-step reasoning:

1. **Understand the Question**: Break down what is being asked
2. **Identify Key Concepts**: Determine the main concepts that need to be addressed
3. **Analyze the Context**: Review the provided source material systematically
4. **Synthesize Information**: Combine information from multiple sources if needed
5. **Verify Your Answer**: Check that your response fully addresses the question
6. **Cite Sources**: Ensure all claims are backed by specific page citations

"""

        # Add query-type-specific instructions
        if query_type == "analytical":
            base_cot += """
**Analytical Query Guidelines**:
- Break down complex concepts into understandable components
- Explain the reasoning behind processes or phenomena
- Show how different elements relate to each other
- Use evidence from the source material to support your analysis
- Consider multiple perspectives if relevant
- When asked for implementations or code examples, translate the analytical concepts into practical code
- Provide code examples that demonstrate how the concepts work in practice
"""
        elif query_type == "comparative":
            base_cot += """
**Comparative Query Guidelines**:
- Clearly identify what is being compared
- Systematically address similarities first, then differences
- Use parallel structure to make comparisons clear
- Provide specific examples or evidence for each point of comparison
- Conclude with the most significant distinctions
"""
        elif query_type == "follow-up":
            base_cot += """
**Follow-Up Query Guidelines**:
- Reference previous context appropriately
- Build upon information already provided
- Add new insights or clarifications
- Maintain consistency with earlier responses
- Address the specific aspect being questioned
"""
        else:  # factual or clarification
            base_cot += """
**Response Guidelines**:
- Provide direct, accurate answers based on the source material
- Define technical terms when they first appear
- Use clear, concise language
- Support all factual claims with page citations
- Organize information logically
"""

        return base_cot


# Trimmed prompt used ONLY for token-budget estimation. The two chat entry
# points previously each kept their own identical copy of this string; it is
# centralized here so they cannot drift apart. It is shorter than the full
# build_system_prompt() output (no few-shot / CoT sections), which is the
# same approximation the call sites already relied on.
TOKEN_BUDGET_TEMPLATE = """You are an AI tutor helping a student understand a PDF document.
You have access to the following document chunks that are relevant to the student's question:

{context_text}

When referring to content, always cite the page number like [Page X].
Make sure to use the correct page number for each piece of information.

IMPORTANT FORMATTING INSTRUCTIONS:
1. Use markdown to highlight important concepts, terms, or phrases by making them **bold** or using *italics*.
2. For direct quotes from the document, use > blockquote formatting.
3. When referring to specific sections, use [Page X] to cite the page number.
4. Use bullet points or numbered lists for step-by-step explanations.
5. For critical information or warnings, use "⚠️" at the beginning of the paragraph.

PDF ANNOTATION FEATURE - IMPORTANT:
You MUST identify specific parts of the document that are relevant to your answer.
At the END of your response, ALWAYS include an ANNOTATIONS section with the following JSON format:

```annotations
[
  {{
    "pageNumber": <page number from context above>,
    "type": "highlight",
    "textToHighlight": "<3-10 word phrase copied exactly from the document>",
    "explanation": "<why this text answers the question>"
  }}
]
```

IMAGE ANNOTATION FORMAT:
```annotations
[
  {{
    "pageNumber": <page number from context above>,
    "type": "circle",
    "imageChunkId": "<Image ID from context>",
    "bbox": [x0, y0, x1, y1],
    "explanation": "<why this image answers the question>"
  }}
]
```

ANNOTATION RULES - FOLLOW STRICTLY:
1. ALWAYS include at least 1 annotation when you reference document content
2. The "pageNumber" MUST match a page number from the [Page X] citations above
3. For TEXT: "textToHighlight" MUST be a short phrase (3-10 words) that appears EXACTLY in the document chunks above
4. For IMAGES: Copy the bbox coordinates and imageChunkId exactly from the context above
5. Use type "highlight" for text (most common), "circle" for images/diagrams, "box" for tables
6. Copy the exact words from the document - do not paraphrase
7. Include 1-3 annotations per response

Make your responses helpful, clear, and educational. If the context doesn't contain the answer,
say you don't have enough information from the document and suggest looking at other pages."""


# Process-wide singleton. PromptBuilder holds no instance state; the singleton
# mirrors the get_annotation_service() convention.
_service: PromptBuilder | None = None


def get_prompt_builder() -> PromptBuilder:
    """Return the process-wide PromptBuilder singleton."""
    global _service
    if _service is None:
        _service = PromptBuilder()
    return _service
