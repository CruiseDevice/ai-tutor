"""
Test script for Density Calculator Service

Validates the density calculator with different content types:
1. Narrative text (low density)
2. Technical text (medium density)
3. Code/formulas (high density)
4. Tables/structured data (high density)
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.density_calculator_service import DensityCalculatorService


def test_narrative_text():
    """Test with plain narrative text (expected: low density, large chunks)."""
    text = """
    The theory of relativity is a fundamental concept in physics that describes
    the relationship between space and time. Albert Einstein developed two theories
    of relativity: special relativity and general relativity. Special relativity
    deals with objects moving at constant speeds in a straight line, while general
    relativity extends this to include gravity and acceleration. These theories
    revolutionized our understanding of the universe and have been confirmed by
    numerous experimental observations over the past century.
    """

    print("\n" + "="*80)
    print("TEST 1: Narrative Text")
    print("="*80)
    print(f"Text sample: {text[:100]}...")

    calculator = DensityCalculatorService()
    metrics = calculator.calculate_density(text)

    print(f"\nDensity Metrics:")
    print(f"  Token Density: {metrics.token_density:.3f}")
    print(f"  Punctuation Density: {metrics.punctuation_density:.3f}")
    print(f"  Special Char Density: {metrics.special_char_density:.3f}")
    print(f"  Numeric Density: {metrics.numeric_density:.3f}")
    print(f"  Line Break Density: {metrics.line_break_density:.3f}")
    print(f"  Whitespace Density: {metrics.whitespace_density:.3f}")
    print(f"\nOverall Density: {metrics.overall_density:.3f}")
    print(f"Content Type: {metrics.content_type}")
    print(f"Recommended Chunk Size: {metrics.recommended_chunk_size} chars")
    print(f"Recommended Overlap: {metrics.recommended_overlap} chars")

    # Assertions
    assert metrics.content_type == 'narrative', f"Expected 'narrative', got '{metrics.content_type}'"
    assert metrics.overall_density < 0.4, f"Expected low density (<0.4), got {metrics.overall_density:.3f}"
    assert metrics.recommended_chunk_size > 1200, f"Expected large chunks (>1200), got {metrics.recommended_chunk_size}"
    print("\n✓ Test passed: Narrative text correctly identified with low density")


def test_technical_text():
    """Test with technical text containing formulas (expected: medium-high density)."""
    text = """
    The quadratic formula x = (-b ± √(b²-4ac)) / 2a solves equations of the form
    ax² + bx + c = 0. The discriminant Δ = b²-4ac determines the nature of the roots:
    - If Δ > 0: two distinct real roots
    - If Δ = 0: one repeated real root
    - If Δ < 0: two complex conjugate roots

    For example, solving 2x² + 5x - 3 = 0:
    a = 2, b = 5, c = -3
    Δ = 5² - 4(2)(-3) = 25 + 24 = 49
    x = (-5 ± √49) / 4 = (-5 ± 7) / 4
    Therefore x = 0.5 or x = -3
    """

    print("\n" + "="*80)
    print("TEST 2: Technical Text (Math Formulas)")
    print("="*80)
    print(f"Text sample: {text[:100]}...")

    calculator = DensityCalculatorService()
    metrics = calculator.calculate_density(text)

    print(f"\nDensity Metrics:")
    print(f"  Token Density: {metrics.token_density:.3f}")
    print(f"  Punctuation Density: {metrics.punctuation_density:.3f}")
    print(f"  Special Char Density: {metrics.special_char_density:.3f}")
    print(f"  Numeric Density: {metrics.numeric_density:.3f}")
    print(f"  Line Break Density: {metrics.line_break_density:.3f}")
    print(f"  Whitespace Density: {metrics.whitespace_density:.3f}")
    print(f"\nOverall Density: {metrics.overall_density:.3f}")
    print(f"Content Type: {metrics.content_type}")
    print(f"Recommended Chunk Size: {metrics.recommended_chunk_size} chars")
    print(f"Recommended Overlap: {metrics.recommended_overlap} chars")

    # Assertions
    assert metrics.content_type in ['technical', 'structured'], f"Expected 'technical' or 'structured', got '{metrics.content_type}'"
    assert metrics.special_char_density > 0.1, f"Expected high special char density (>0.1), got {metrics.special_char_density:.3f}"
    assert metrics.numeric_density > 0.15, f"Expected high numeric density (>0.15), got {metrics.numeric_density:.3f}"
    print("\n✓ Test passed: Technical text correctly identified with medium-high density")


def test_code_snippet():
    """Test with code snippet (expected: high density, small chunks)."""
    text = """
    def calculate_distance(x1, y1, x2, y2):
        \"\"\"Calculate Euclidean distance between two points.\"\"\"
        dx = x2 - x1
        dy = y2 - y1
        return (dx**2 + dy**2)**0.5

    # Example usage
    points = [(0, 0), (3, 4), (6, 8)]
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        dist = calculate_distance(p1[0], p1[1], p2[0], p2[1])
        print(f"Distance from {p1} to {p2}: {dist:.2f}")
    """

    print("\n" + "="*80)
    print("TEST 3: Code Snippet")
    print("="*80)
    print(f"Text sample: {text[:100]}...")

    calculator = DensityCalculatorService()
    metrics = calculator.calculate_density(text)

    print(f"\nDensity Metrics:")
    print(f"  Token Density: {metrics.token_density:.3f}")
    print(f"  Punctuation Density: {metrics.punctuation_density:.3f}")
    print(f"  Special Char Density: {metrics.special_char_density:.3f}")
    print(f"  Numeric Density: {metrics.numeric_density:.3f}")
    print(f"  Line Break Density: {metrics.line_break_density:.3f}")
    print(f"  Whitespace Density: {metrics.whitespace_density:.3f}")
    print(f"\nOverall Density: {metrics.overall_density:.3f}")
    print(f"Content Type: {metrics.content_type}")
    print(f"Recommended Chunk Size: {metrics.recommended_chunk_size} chars")
    print(f"Recommended Overlap: {metrics.recommended_overlap} chars")

    # Assertions
    assert metrics.content_type == 'structured', f"Expected 'structured', got '{metrics.content_type}'"
    assert metrics.overall_density > 0.5, f"Expected high density (>0.5), got {metrics.overall_density:.3f}"
    assert metrics.recommended_chunk_size < 1000, f"Expected small chunks (<1000), got {metrics.recommended_chunk_size}"
    print("\n✓ Test passed: Code snippet correctly identified with high density")


def test_table_data():
    """Test with table/structured data (expected: very high density, smallest chunks)."""
    text = """
    | Student ID | Name          | Grade | Score | Rank |
    |------------|---------------|-------|-------|------|
    | 12345      | Alice Johnson | A+    | 98.5  | 1    |
    | 12346      | Bob Smith     | A     | 92.3  | 2    |
    | 12347      | Carol Davis   | A-    | 89.7  | 3    |
    | 12348      | David Lee     | B+    | 87.2  | 4    |
    | 12349      | Emma Wilson   | B     | 84.1  | 5    |
    | 12350      | Frank Brown   | B-    | 81.5  | 6    |

    Summary Statistics:
    - Mean: 88.9
    - Median: 88.45
    - Std Dev: 6.2
    - Pass Rate: 100%
    """

    print("\n" + "="*80)
    print("TEST 4: Table/Structured Data")
    print("="*80)
    print(f"Text sample: {text[:100]}...")

    calculator = DensityCalculatorService()
    metrics = calculator.calculate_density(text)

    print(f"\nDensity Metrics:")
    print(f"  Token Density: {metrics.token_density:.3f}")
    print(f"  Punctuation Density: {metrics.punctuation_density:.3f}")
    print(f"  Special Char Density: {metrics.special_char_density:.3f}")
    print(f"  Numeric Density: {metrics.numeric_density:.3f}")
    print(f"  Line Break Density: {metrics.line_break_density:.3f}")
    print(f"  Whitespace Density: {metrics.whitespace_density:.3f}")
    print(f"\nOverall Density: {metrics.overall_density:.3f}")
    print(f"Content Type: {metrics.content_type}")
    print(f"Recommended Chunk Size: {metrics.recommended_chunk_size} chars")
    print(f"Recommended Overlap: {metrics.recommended_overlap} chars")

    # Assertions
    assert metrics.content_type == 'structured', f"Expected 'structured', got '{metrics.content_type}'"
    assert metrics.overall_density > 0.6, f"Expected very high density (>0.6), got {metrics.overall_density:.3f}"
    assert metrics.recommended_chunk_size <= 800, f"Expected very small chunks (<=800), got {metrics.recommended_chunk_size}"
    print("\n✓ Test passed: Table data correctly identified with very high density")


def test_empty_text():
    """Test with empty text (edge case)."""
    text = ""

    print("\n" + "="*80)
    print("TEST 5: Empty Text (Edge Case)")
    print("="*80)

    calculator = DensityCalculatorService()
    metrics = calculator.calculate_density(text)

    print(f"\nDensity Metrics:")
    print(f"  Overall Density: {metrics.overall_density:.3f}")
    print(f"  Content Type: {metrics.content_type}")
    print(f"  Recommended Chunk Size: {metrics.recommended_chunk_size} chars")

    # Assertions
    assert metrics.content_type == 'empty', f"Expected 'empty', got '{metrics.content_type}'"
    assert metrics.overall_density == 0.0, f"Expected zero density, got {metrics.overall_density:.3f}"
    print("\n✓ Test passed: Empty text handled correctly")


def main():
    """Run all density calculator tests."""
    print("\n" + "="*80)
    print("DENSITY CALCULATOR SERVICE - VALIDATION TESTS")
    print("="*80)

    try:
        test_narrative_text()
        test_technical_text()
        test_code_snippet()
        test_table_data()
        test_empty_text()

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nSummary:")
        print("  ✓ Narrative text: Low density, large chunks")
        print("  ✓ Technical text: Medium-high density, standard chunks")
        print("  ✓ Code snippets: High density, small chunks")
        print("  ✓ Table data: Very high density, smallest chunks")
        print("  ✓ Empty text: Edge case handled correctly")
        print("\nDensity Calculator Service is working correctly!")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
