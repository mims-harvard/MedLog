def format_rich(value: str, markup: str) -> str:
    """Format string with rich markup.

    Args:
        value: The string to format.
        markup: The rich markup to apply.

    Returns:
        The formatted string.
    """
    return f"[{markup}]{value}[/{markup}]"
