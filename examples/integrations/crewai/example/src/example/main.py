#!/usr/bin/env python
# import warnings


from example.crew import Example

# warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    try:
        Example().crew().kickoff(inputs={"location": "New York City"})
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
