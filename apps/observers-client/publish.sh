#!/bin/sh
# Build the web page and publish it to the `odeon-client` repository, whose
# working tree is `dist/` and whose git directory is `odeon-client.git`
# next to it (trunk empties `dist/` at every build, so the history cannot
# live inside it). Run from anywhere:
#
#     apps/observers-client/publish.sh            # build for /odeon-client/, commit, push
#     apps/observers-client/publish.sh /          # another public path
#
# One-time setup (creates the repository on GitHub and enables Pages):
#
#     cd apps/observers-client
#     git --git-dir=odeon-client.git --work-tree=dist init -b main
#     git --git-dir=odeon-client.git config core.worktree "$PWD/dist"
#     gh repo create odeon-client --public
#     git --git-dir=odeon-client.git remote add origin https://github.com/<user>/odeon-client.git
#     ./publish.sh
#     gh api -X POST repos/<user>/odeon-client/pages -f 'source[branch]=main' -f 'source[path]=/'
set -e
cd "$(dirname "$0")"
public_url="${1:-/odeon-client/}"
git_dir="$PWD/odeon-client.git"
[ -d "$git_dir" ] || { echo "no $git_dir: do the one-time setup at the top of this script" >&2; exit 1; }

trunk build --public-url "$public_url"

git --git-dir="$git_dir" add -A
if git --git-dir="$git_dir" diff --cached --quiet; then
    echo "page unchanged, nothing to publish"
    exit 0
fi
git --git-dir="$git_dir" commit -q -m "page build $(date '+%Y-%m-%d %H:%M') (public url $public_url)"
git --git-dir="$git_dir" push -u origin main
