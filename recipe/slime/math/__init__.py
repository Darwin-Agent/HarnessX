"""Math RL task — MathBoxedEvaluator + retool reward shaping + code_interpreter.

All math task domain code lives here:
    builder.py    — MathTaskBuilder (dapo-math-17k + AIME sample formats)
    evaluator.py  — MathBoxedEvaluator (\\boxed{} answer evaluation)
    rewards.py    — RetoolCompatPRM + math_format_reward
    tools.py      — code_interpreter_tool (Python sandbox, math modules)
    formatter.py  — retool Jinja2 tokenization formatters
    data_prep.py  — dataset download + preprocessing scripts
"""
