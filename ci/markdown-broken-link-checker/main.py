import argparse
import asyncio
import glob
import os
import sys
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import httpx
import markdown
from bs4 import BeautifulSoup
from tqdm import tqdm

# Links that should not be checked (e.g. anti-bot measures)
WHITELIST = {
    "https://www.anthropic.com/engineering/claude-think-tool",
    "https://www.x.com/tensorzero",
}

# Folders that should not be traversed
BLACKLISTED_FOLDERS = {"node_modules"}


async def check_link(client: httpx.AsyncClient, link: str, origin_files: List[str]) -> Tuple[List[str], str, int, str]:
    """Check if a link is broken."""
    try:
        # Skip checking links that are whitelisted
        if link in WHITELIST:
            return origin_files, link, 0, "Skipped (whitelisted)"

        # Skip checking links that are not http/https
        parsed_url = urlparse(link)
        if parsed_url.scheme not in ("http", "https"):
            return origin_files, link, 0, "Skipped (not HTTP/HTTPS)"

        response = await client.get(link, follow_redirects=True)
        return (
            origin_files,
            link,
            response.status_code,
            "OK" if response.status_code < 400 else "Broken",
        )
    except httpx.RequestError as e:
        return origin_files, link, 0, f"Error: {str(e)}"
    except Exception as e:
        return origin_files, link, 0, f"Error: {str(e)}"


def extract_links_from_md(file_path: str) -> List[str]:
    """Extract all links from a markdown file using a proper parser."""
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Convert markdown to HTML
    html = markdown.markdown(md_content)

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Extract all links
    links = []
    for a_tag in soup.find_all("a", href=True):
        links.append(a_tag["href"])

    return links


def is_blacklisted(file_path: str) -> bool:
    """Check if the file path contains a blacklisted folder."""
    for folder in BLACKLISTED_FOLDERS:
        if f"/{folder}/" in file_path or file_path.startswith(f"{folder}/"):
            return True
    return False


async def check_files(path: str) -> List[Tuple[List[str], str, int, str]]:
    """Check all markdown files in the given path for broken links."""
    all_md_files = glob.glob(os.path.join(path, "**", "*.md"), recursive=True)

    # Filter out files in blacklisted folders
    md_files = [f for f in all_md_files if not is_blacklisted(f)]
    skipped_files = len(all_md_files) - len(md_files)

    if skipped_files > 0:
        print(f"Skipped {skipped_files} files in blacklisted folders")

    if not md_files:
        print(f"No markdown files found in {path} (outside of blacklisted folders)")
        return []

    print(f"Found {len(md_files)} markdown files to check")

    # Collect all links first with progress bar
    # Use a dictionary to track which files contain each link
    link_to_files: Dict[str, List[str]] = {}
    total_links_found = 0

    for md_file in tqdm(md_files, desc="Scanning files", unit="file"):
        links = extract_links_from_md(md_file)
        total_links_found += len(links)
        for link in links:
            if link not in link_to_files:
                link_to_files[link] = []
            link_to_files[link].append(md_file)

    print(f"Found {total_links_found} total links ({len(link_to_files)} unique) to check")

    # Check all links with progress bar
    results = []
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    timeout = httpx.Timeout(10.0, connect=5.0)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        # Create tasks for all unique links
        tasks = [check_link(client, link, files) for link, files in link_to_files.items()]

        # Process tasks with progress bar
        if tasks:
            for future in tqdm(
                asyncio.as_completed(tasks),
                desc="Checking links",
                total=len(tasks),
                unit="link",
            ):
                result = await future
                results.append(result)
        else:
            print("No links found in markdown files")

    return results


def main():
    parser = argparse.ArgumentParser(description="Check for broken links in markdown files")
    parser.add_argument("path", help="Path to search for markdown files")
    parser.add_argument("--whitelist", "-w", nargs="+", help="Additional URLs to whitelist")
    parser.add_argument(
        "--blacklist-folders",
        "-b",
        nargs="+",
        help="Additional folders to skip (already skips node_modules)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Path does not exist: {args.path}")
        sys.exit(1)

    # Add any additional whitelist URLs
    if args.whitelist:
        for url in args.whitelist:
            WHITELIST.add(url)

    # Add any additional blacklisted folders
    if args.blacklist_folders:
        for folder in args.blacklist_folders:
            BLACKLISTED_FOLDERS.add(folder)

    if WHITELIST:
        print(f"Whitelisted {len(WHITELIST)} URLs that will not be checked")

    if BLACKLISTED_FOLDERS:
        print(f"Blacklisted folders that will be skipped: {', '.join(BLACKLISTED_FOLDERS)}")

    print(f"Checking markdown files in {args.path}")

    results = asyncio.run(check_files(args.path))

    # Format and display results
    if results:
        broken_links = [r for r in results if r[3] != "OK" and not r[3].startswith("Skipped")]
        skipped_links = [r for r in results if r[3].startswith("Skipped")]
        whitelisted_links = [r for r in skipped_links if "whitelisted" in r[3]]

        print("\nSummary:")
        print(f"Total unique links checked: {len(results)}")
        print(f"Valid links: {len(results) - len(broken_links) - len(skipped_links)}")
        print(f"Broken links: {len(broken_links)}")
        print(f"Skipped links: {len(skipped_links)}")
        print(f"  - Whitelisted: {len(whitelisted_links)}")
        print(f"  - Non-HTTP/HTTPS: {len(skipped_links) - len(whitelisted_links)}")
        print(f"Blacklisted folders: {len(BLACKLISTED_FOLDERS)}")

        if broken_links:
            print("\nBroken Links:")
            for files, link, status, msg in broken_links:
                print(f"Link: {link}")
                print(f"Status: {status}")
                print(f"Message: {msg}")
                print(f"Found in {len(files)} files:")
                for file in files:
                    print(f"  - {file}")
                print("-" * 40)

            # Exit with error if broken links are found
            sys.exit(1)

    print("All links are valid!")


if __name__ == "__main__":
    main()
