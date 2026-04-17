# GitHub setup and LaTeX sync guide

This guide gets the repository onto GitHub and connects it to your LaTeX workflow (Overleaf or local).

## 1. One-time GitHub setup

### Step 1a — Create the private repo on GitHub

1. Go to https://github.com/new while signed in as **Cyrus44511**.
2. Set the **repository name** to: `hank-edgeworth-complementarity`
3. Set **visibility** to **Private**.
4. **Do not** initialize with README, .gitignore, or license (we already have these).
5. Click **Create repository**.

GitHub will show a page with setup instructions. Ignore them — use the commands below instead.

### Step 1b — Push the local repo to GitHub

Open a terminal and run:

```bash
cd "/Users/brigh/Documents/Claude/Projects/HANK JOHN/hank-edgeworth-complementarity"

# If git isn't configured yet with your identity, run these first:
git config --global user.name  "Bright Quaye"
git config --global user.email "b.quaye@wustl.edu"

# Initialize and commit (skip if already done):
git init -b main
git add .
git commit -m "Initial import: paper draft, code, reproduction pipeline"

# Add the GitHub remote:
git remote add origin https://github.com/Cyrus44511/hank-edgeworth-complementarity.git

# Push:
git push -u origin main
```

GitHub will prompt you for credentials. You have two options:

**Option A — HTTPS with a personal access token (recommended):**

1. Go to https://github.com/settings/tokens?type=beta
2. Click **Generate new token** → set expiry, select only the `hank-edgeworth-complementarity` repo, grant **Contents: Read and write**.
3. Copy the token. When git asks for your password, paste the token.

**Option B — SSH:**

```bash
# Generate a key if you don't have one:
ssh-keygen -t ed25519 -C "b.quaye@wustl.edu"

# Add it to ssh-agent:
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy the public key:
cat ~/.ssh/id_ed25519.pub
# (paste it at https://github.com/settings/ssh/new)

# Then switch the remote:
git remote set-url origin git@github.com:Cyrus44511/hank-edgeworth-complementarity.git
git push -u origin main
```

## 2. Connect to LaTeX

You have three workflows. Pick one.

### Workflow A — Overleaf (recommended, easiest cloud sync)

1. On GitHub, go to the repo you just pushed.
2. On Overleaf, click **New Project** → **Import from GitHub**. (This requires an Overleaf **paid** account — free accounts can only import via ZIP once.)
3. Select `Cyrus44511/hank-edgeworth-complementarity`.
4. Set the **Main document** to `paper/main.tex`.
5. You're done — Overleaf will pull updates whenever you click **Sync → GitHub**.

If you have a free Overleaf account:
1. From Overleaf: **New Project → Upload Project → ZIP** and upload a zip of the repo.
2. Then clone your Overleaf project locally and link that copy to GitHub. Overleaf's Git bridge is available in paid plans only.

### Workflow B — Local LaTeX (VS Code, TeXShop, TeXStudio, MacTeX)

1. Clone the repo once:
   ```bash
   cd ~/Documents
   git clone https://github.com/Cyrus44511/hank-edgeworth-complementarity.git
   ```
2. Open `paper/main.tex` in your editor.
3. Build with `latexmk -pdf paper/main.tex` or your editor's compile command.
4. Push your changes:
   ```bash
   cd ~/Documents/hank-edgeworth-complementarity
   git add -A
   git commit -m "Describe your change"
   git push
   ```

### Workflow C — Keep working in this folder and sync from here

The folder `/Users/brigh/Documents/Claude/Projects/HANK JOHN/hank-edgeworth-complementarity` is already the git repo. Whenever you or Claude change files:

```bash
cd "/Users/brigh/Documents/Claude/Projects/HANK JOHN/hank-edgeworth-complementarity"
git add -A
git commit -m "Describe change"
git push
```

## 3. Compiling the paper

You need a LaTeX distribution:

- **macOS**: MacTeX — https://www.tug.org/mactex/
- **Linux**: `sudo apt-get install texlive-full`
- **Overleaf**: nothing to install

Packages required (all in TeX Live / MacTeX):
`geometry`, `setspace`, `amsmath`, `amssymb`, `amsthm`, `mathtools`, `bm`, `graphicx`, `booktabs`, `array`, `tabularx`, `multirow`, `threeparttable`, `siunitx`, `caption`, `enumitem`, `hyperref`, `cleveref`, `natbib`, `microtype`, `lmodern`.

Build command:
```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Or with `latexmk`:
```bash
cd paper
latexmk -pdf main
```

## 4. Keeping code and paper in sync

Whenever you re-run the simulations:

```bash
python3 code/run_all.py     # regenerates paper/figures/*
cd paper && latexmk -pdf main
git add paper/figures paper/main.pdf
git commit -m "Update figures and compiled PDF"
git push
```

## 5. Inviting collaborators

In the private GitHub repo, go to **Settings → Collaborators → Add people**.

## Troubleshooting

- `fatal: remote origin already exists` → `git remote set-url origin <URL>`
- `push rejected, non-fast-forward` → `git pull --rebase` then push
- Missing LaTeX package errors → install via `tlmgr install <package>` (MacTeX)
- Can't find `natbib` → install `texlive-publishers` on Linux
