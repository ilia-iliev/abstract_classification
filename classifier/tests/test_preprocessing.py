from classifier.preprocessing import FORMULA_TOKEN, prepare_abstract


def test_latex_flattens_short_inline_math_and_removes_formatting():
    text = r"An \emph{important} result: $\alpha + \beta$ with \nu=1."

    assert prepare_abstract(text) == "An important result: alpha + beta with \\nu=1."


def test_latex_removes_declaration_style_formatting():
    assert prepare_abstract(r"The {\it important result} holds.") == "The important result holds."


def test_latex_flattens_subscripts_and_numbers():
    text = r"H$_2$O and CO$_2$ in a $2$-dimensional model."

    assert prepare_abstract(text) == "H2O and CO2 in a 2-dimensional model."
    assert prepare_abstract(r"$f:\{0,1\}^n\to\{0,1\}^n$") == "f:0,1^n to 0,1^n"


def test_latex_preserves_percentages_and_text_after_them():
    text = "The error was 3%.\nTo calibrate, we moved the sample."

    assert prepare_abstract(text) == "The error was 3%. To calibrate, we moved the sample."


def test_latex_masks_long_inline_math():
    text = r"Result $abcdefghijklmnopqrstuvwxyz$ follows."

    assert prepare_abstract(text) == f"Result {FORMULA_TOKEN} follows."


def test_latex_preserves_unbalanced_math_and_unknown_macros():
    text = r"Broken $x with \textit{style} and \alpha."

    assert prepare_abstract(text) == text


def test_latex_masks_balanced_math_environment():
    text = r"A \begin{equation} x = y \end{equation} result."

    assert prepare_abstract(text) == f"A {FORMULA_TOKEN} result."
