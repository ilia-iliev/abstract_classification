import re

from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexCommentNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexWalker,
)

MAX_INPUT_LENGTH = 512
# Kept as an alias while callers migrate to the context-length terminology.
MAX_CONTEXT_LENGTH = MAX_INPUT_LENGTH
MAX_INLINE_MATH_LENGTH = 24
FORMULA_TOKEN = "<FORMULA>"
PRIMARY_LABEL_TARGET = 1.0
SECONDARY_LABEL_TARGET = 0.5
PREPROCESSING_VERSION = "inline_math_24_formula_token_v2"


def register_formula_token(tokenizer):
    """Make the formula placeholder one indivisible, persisted tokenizer token."""
    tokenizer.add_special_tokens({"additional_special_tokens": [FORMULA_TOKEN]})
    token_id = tokenizer.convert_tokens_to_ids(FORMULA_TOKEN)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer did not register {FORMULA_TOKEN}")
    return token_id
_WHITESPACE_PATTERN = re.compile(r"\s+")
_TEX_COMMAND_PATTERN = re.compile(r"\\([A-Za-z]+)\*?")
_FORMATTING_MACROS = frozenset({"emph", "textbf", "textit", "textrm", "textsc", "texttt", "underline"})
_DECLARATION_FORMATTING_MACROS = frozenset({"bf", "it", "rm", "sc", "sf", "sl", "tt"})
_MATH_ENVIRONMENTS = frozenset({
    "align",
    "align*",
    "alignat",
    "alignat*",
    "displaymath",
    "equation",
    "equation*",
    "gather",
    "gather*",
    "math",
    "multline",
    "multline*",
})
_COMMAND_TEXT = {
    "cdot": " ",
    "ge": " greater than or equal to ",
    "geq": " greater than or equal to ",
    "le": " less than or equal to ",
    "leq": " less than or equal to ",
    "mp": " plus or minus ",
    "pm": " plus or minus ",
    "propto": " proportional to ",
    "sim": " approximately ",
    "times": " x ",
    "to": " to ",
}
_IGNORED_COMMANDS = frozenset({",", ";", "!", "quad", "qquad"})


def _source(text, node):
    return text[node.pos:node.pos + node.len]


def _flatten_inline_math(source):
    value = source.strip()
    value = re.sub(r"\\([{}%_#$&])", r"\1", value)
    value = re.sub(r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|textrm|text)\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", value)

    def command_text(match):
        command = match.group(1)
        if command in _IGNORED_COMMANDS:
            return " "
        return _COMMAND_TEXT.get(command, f" {command} ")

    value = _TEX_COMMAND_PATTERN.sub(command_text, value)
    value = value.replace("_", "").replace("{", "").replace("}", "")
    return value


def _render_latex_nodes(text, nodes):
    rendered = []
    for node in nodes:
        if isinstance(node, LatexCharsNode):
            rendered.append(node.chars)
        elif isinstance(node, LatexCommentNode):
            rendered.append(_source(text, node))
        elif isinstance(node, LatexGroupNode):
            rendered.append(_render_latex_nodes(text, node.nodelist))
        elif isinstance(node, LatexMathNode):
            source = _source(text, node)
            if not source.endswith(node.delimiters[1]):
                rendered.append(source)
                continue
            content = source[len(node.delimiters[0]):-len(node.delimiters[1])]
            if node.displaytype == "inline" and len(content.strip()) < MAX_INLINE_MATH_LENGTH:
                rendered.append(_flatten_inline_math(content))
            else:
                rendered.append(f" {FORMULA_TOKEN} ")
        elif isinstance(node, LatexEnvironmentNode):
            source = _source(text, node)
            end_marker = f"\\end{{{node.environmentname}}}"
            if node.environmentname in _MATH_ENVIRONMENTS and source.endswith(end_marker):
                rendered.append(f" {FORMULA_TOKEN} ")
            else:
                rendered.append(source)
        elif isinstance(node, LatexMacroNode) and node.macroname in _DECLARATION_FORMATTING_MACROS:
            continue
        elif isinstance(node, LatexMacroNode) and node.macroname in _FORMATTING_MACROS:
            arguments = node.nodeargd.argnlist
            if len(arguments) == 1 and isinstance(arguments[0], LatexGroupNode):
                rendered.append(_render_latex_nodes(text, arguments[0].nodelist))
            else:
                rendered.append(_source(text, node))
        else:
            rendered.append(_source(text, node))
    return "".join(rendered)


def prepare_abstract(text):
    """Remove formatting, flatten short inline math, and mask long formulas with a token."""
    nodes, _, _ = LatexWalker(text).get_latex_nodes()
    converted = _render_latex_nodes(text, nodes)
    return _WHITESPACE_PATTERN.sub(" ", converted).strip()
