"""Minimal command-line entry point for SmartGroceryAI.

Python runs this module when ``python -m app`` is executed. Keeping an entry point
available from the beginning provides a quick way to verify that the package and
its environment are installed correctly before a web interface is introduced.
"""


def main() -> None:
    """Run the current application placeholder.

    The output is deliberately simple. A later application server can replace this
    behavior while preserving ``main`` as a clear startup boundary.
    """
    print("SmartGroceryAI")


if __name__ == "__main__":
    main()
