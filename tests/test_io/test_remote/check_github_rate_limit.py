"""
Utility script to check GitHub API rate limit status.

Run this to see if GitHub remote tests can be executed.
"""

from __future__ import annotations

import sys
from datetime import datetime


def check_rate_limit():
    """Check and display GitHub API rate limit status."""
    import os

    try:
        import requests
    except ImportError:
        print("❌ requests library not installed")
        print("Install with: pip install requests")
        return False

    # Check if token is set (check in priority order)
    token = (
        os.getenv("DASDAE_GITHUB_TOKEN")
        or os.getenv("GITHUB_TOKEN")
        or os.getenv("GH_TOKEN")
    )

    try:
        headers = {}
        if token:
            headers["Authorization"] = f"token {token}"

        response = requests.get(
            "https://api.github.com/rate_limit", headers=headers, timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            core = data["resources"]["core"]

            print("=" * 60)
            print("GitHub API Rate Limit Status")
            print("=" * 60)
            token_status = "✓ Using token" if token else "No token (unauthenticated)"
            print(f"Authentication: {token_status}")
            print(f"Remaining requests: {core['remaining']}/{core['limit']}")

            if core["remaining"] == 0:
                reset_time = datetime.fromtimestamp(core["reset"])
                now = datetime.now()
                time_left = reset_time - now
                minutes = int(time_left.total_seconds() / 60)

                print(f"\n❌ STATUS: RATE LIMITED")
                print(f"   Reset in: {minutes} minutes")
                print(f"   Reset at: {reset_time.strftime('%H:%M:%S')}")
                print("\nGitHub remote tests will be SKIPPED until rate limit resets.")
                print("\nTo avoid rate limiting:")
                if not token:
                    print("  1. Set a GitHub token:")
                    print("     export DASDAE_GITHUB_TOKEN=your_token_here")
                    print("     (Create token at: github.com/settings/tokens)")
                print("  2. Wait for the rate limit to reset")
                print("  3. Run fewer tests at once")
                return False
            elif core["remaining"] < 10:
                print(f"\n⚠️  WARNING: Low rate limit ({core['remaining']} remaining)")
                print("   GitHub tests may fail if limit is exceeded")
                return True
            else:
                print("\n✓ STATUS: Available")
                print("  GitHub remote tests can run successfully")
                return True
        else:
            print(f"❌ Could not check rate limit: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking rate limit: {e}")
        return False


if __name__ == "__main__":
    can_run = check_rate_limit()
    sys.exit(0 if can_run else 1)
