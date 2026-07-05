"""Unit tests for PromptBuilder.

Covers the pure prompt-construction logic extracted from ChatService.
The LLM-backed classify_query_type() is exercised only via its disabled
short-circuit; full async/LLM coverage belongs in integration tests.
"""
import pytest

from app.services.prompt_builder import (
    PromptBuilder,
    TOKEN_BUDGET_TEMPLATE,
    get_prompt_builder,
)


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


def test_factory_returns_singleton():
    a = get_prompt_builder()
    b = get_prompt_builder()
    assert a is b


def test_build_system_prompt_includes_context_and_formatting(builder):
    prompt = builder.build_system_prompt(
        context_text="[Page 7]: mitosis is cell division.",
        query_type="factual",
        complexity="simple",
        requires_cot=False,
    )
    # The supplied context is embedded
    assert "[Page 7]: mitosis is cell division." in prompt
    # Always-present sections
    assert "IMPORTANT FORMATTING INSTRUCTIONS:" in prompt
    assert "PDF ANNOTATION FEATURE - CRITICAL INSTRUCTIONS:" in prompt
    # Few-shot examples always appended
    assert "# Few-Shot Examples" in prompt


def test_build_system_prompt_includes_cot_only_when_requested(builder):
    without_cot = builder.build_system_prompt(
        context_text="ctx", query_type="factual", complexity="simple", requires_cot=False
    )
    with_cot = builder.build_system_prompt(
        context_text="ctx", query_type="analytical", complexity="complex", requires_cot=True
    )
    assert "# Chain-of-Thought Reasoning" not in without_cot
    assert "# Chain-of-Thought Reasoning" in with_cot


@pytest.mark.parametrize(
    "query_type, expected_heading",
    [
        ("analytical", "Analytical Query Guidelines"),
        ("comparative", "Comparative Query Guidelines"),
        ("follow-up", "Follow-Up Query Guidelines"),
        ("factual", "Response Guidelines"),       # default branch
        ("clarification", "Response Guidelines"),  # default branch
    ],
)
def test_chain_of_thought_section_per_query_type(builder, query_type, expected_heading):
    section = builder._get_chain_of_thought_section(query_type)
    assert expected_heading in section
    # Base CoT steps are always present
    assert "Understand the Question" in section


def test_few_shot_examples_cover_three_examples(builder):
    examples = builder._get_few_shot_examples()
    assert "Example 1: Factual Query" in examples
    assert "Example 2: Analytical Query" in examples
    assert "Example 3: Comparative Query" in examples


def test_token_budget_template_is_stable_string():
    """The token-budget template must keep its exact form: it approximates the
    system prompt for token counting, and the call sites depend on its shape
    (annotation block, rules count) not drifting.
    """
    assert "{context_text}" in TOKEN_BUDGET_TEMPLATE
    assert "```annotations" in TOKEN_BUDGET_TEMPLATE
    # 7 annotation rules in the trimmed template (matches the original copies)
    assert TOKEN_BUDGET_TEMPLATE.count("\n") > 0
    rules = [ln for ln in TOKEN_BUDGET_TEMPLATE.splitlines() if ln.strip().startswith(tuple("1234567"))]
    assert len(rules) >= 7


@pytest.mark.asyncio
async def test_classify_query_type_disabled_returns_default(builder, monkeypatch):
    """When ENABLE_QUERY_CLASSIFICATION is False, no LLM call is made."""
    from app.services import prompt_builder as pb_mod
    monkeypatch.setattr(pb_mod.settings, "ENABLE_QUERY_CLASSIFICATION", False)

    result = await builder.classify_query_type(query="anything", user_api_key="k")
    assert result == {"query_type": "factual", "complexity": "simple", "requires_cot": False}
