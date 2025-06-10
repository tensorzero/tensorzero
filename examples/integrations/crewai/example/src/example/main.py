#!/usr/bin/env python


from example.crew import Example


def run():
    try:
        Example().crew().kickoff(inputs={"location": "New York City"})
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
