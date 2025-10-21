#!/usr/bin/env python3
"""
Process JUnit XML test output and convert nested flaky/rerun failure elements into retries.

This script reads JUnit XML files, finds test cases with <flakyFailure> or <rerunFailure> elements,
and restructures them to show test retries more clearly. Each retry is converted into a
separate test case with a <failure> tag.
"""

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def process_flaky_tests(input_file, output_file=None):
    """
    Process JUnit XML and convert flakyTest elements into retry test cases.

    Args:
        input_file: Path to input JUnit XML file
        output_file: Path to output file (None = print to stdout)
    """
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        return 1

    # Process all testsuites
    for testsuite in root.findall('.//testsuite'):
        process_testsuite(testsuite)

    # Write output
    if output_file:
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        print(f"Processed XML written to: {output_file}")
    else:
        # Print to stdout
        ET.dump(root)

    return 0


def process_testsuite(testsuite):
    """Process a single testsuite element and handle flaky tests."""
    testcases = list(testsuite.findall('testcase'))
    new_testcases = []

    for testcase in testcases:
        flaky_failures = list(testcase.findall('flakyFailure'))
        rerun_failures = list(testcase.findall('rerunFailure'))
        all_failures = flaky_failures + rerun_failures

        if all_failures:
            # Original test failed, followed by retries
            new_testcases.extend(create_retry_testcases(testcase, all_failures))
        else:
            # Regular test case, keep as is
            new_testcases.append(testcase)

    # Remove old testcases and add new ones
    for testcase in testcases:
        testsuite.remove(testcase)

    for testcase in new_testcases:
        testsuite.append(testcase)

    # Update testsuite counts
    update_testsuite_counts(testsuite, new_testcases)


def create_retry_testcases(original_testcase, failure_elements):
    """
    Create separate test case entries for each retry.

    Args:
        original_testcase: The original testcase element
        failure_elements: List of flakyFailure or rerunFailure elements

    Returns:
        List of testcase elements representing the original and all retries
    """
    retry_testcases = []

    # Convert each flakyFailure/rerunFailure into a separate test run with a <failure> tag
    for failure_element in failure_elements:
        retry_testcase = ET.Element('testcase', original_testcase.attrib)

        # Create a <failure> element from the flakyFailure/rerunFailure attributes and content
        failure = ET.Element('failure')

        # Copy attributes from failure_element to failure
        for attr_name, attr_value in failure_element.attrib.items():
            failure.set(attr_name, attr_value)

        # If message attribute is missing, use type attribute as message
        if 'message' not in failure.attrib and 'type' in failure.attrib:
            failure.set('message', failure.get('type'))

        # Copy text content and children from failure_element
        if failure_element.text:
            failure.text = failure_element.text
        if failure_element.tail:
            failure.tail = failure_element.tail

        for child in failure_element:
            failure.append(child)

        retry_testcase.append(failure)
        retry_testcases.append(retry_testcase)

    # Final successful attempt (if the test eventually passed)
    # The original testcase without flakyFailure/rerunFailure children represents the final state
    final_attempt = ET.Element('testcase', original_testcase.attrib)

    # Copy non-flaky/rerun children (like system-out, system-err)
    for child in original_testcase:
        if child.tag not in ('flakyFailure', 'rerunFailure'):
            final_attempt.append(child)

    retry_testcases.append(final_attempt)

    return retry_testcases


def update_testsuite_counts(testsuite, testcases):
    """Update testsuite test counts based on processed testcases."""
    total_tests = len(testcases)
    failures = sum(1 for tc in testcases if tc.find('failure') is not None)
    errors = sum(1 for tc in testcases if tc.find('error') is not None)
    skipped = sum(1 for tc in testcases if tc.find('skipped') is not None)

    testsuite.set('tests', str(total_tests))
    testsuite.set('failures', str(failures))
    testsuite.set('errors', str(errors))
    if skipped > 0:
        testsuite.set('skipped', str(skipped))


def main():
    parser = argparse.ArgumentParser(
        description='Process JUnit XML and convert flakyFailure/rerunFailure elements into test retries'
    )
    parser.add_argument(
        'input_file',
        type=Path,
        help='Input JUnit XML file'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file (default: stdout)'
    )

    args = parser.parse_args()

    return process_flaky_tests(args.input_file, args.output)


if __name__ == '__main__':
    sys.exit(main())
