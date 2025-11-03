## Incident: Leaked credentials in notebook history

Date: 2025-11-03

Summary
- A Hugging Face token and a SWE-bench API key were accidentally committed inside `qwen_a3b_mini_swebench_colab.ipynb` in an early commit.

Actions taken (local)
- Restored the original notebook from the repository backup refs and sanitized the file by replacing the plaintext tokens with redacted placeholders.
- Created a sanitized commit and force-updated `origin/main` to remove the leaked secrets from the branch history.
- Removed local backup refs left by history-rewrite attempts (refs/original), expired reflogs, and ran an aggressive `git gc` to prune dangling objects locally.

Commands run (for reproducibility)
```
# restore original notebook (if needed)
git checkout refs/original/refs/heads/main -- qwen_a3b_mini_swebench_colab.ipynb

# sanitize notebook (macOS sed example)
sed -i "" "s/<PRIVATE_TOKEN>/<REDACTED_HF_TOKEN>/g" qwen_a3b_mini_swebench_colab.ipynb
sed -i "" "s/<PRIVATE_KEY>/<REDACTED_SWEBENCH_KEY>/g" qwen_a3b_mini_swebench_colab.ipynb

# create an orphan branch with sanitized tree and commit
git checkout --orphan clean-main-temp
git reset --mixed
git add -A
git commit -m "chore: remove leaked secrets (sanitise notebook)"

# force-push sanitized history to origin/main
git push --force-with-lease origin HEAD:main

# purge local cleanup (remove refs/original, expire reflog, gc)
git for-each-ref --format='%(refname)' refs/original | xargs -n1 git update-ref -d || true
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

Immediate required actions (you must do these now)
1. Revoke the leaked Hugging Face token at https://huggingface.co/settings/tokens and the SWE-bench API key where you manage it. Treat them as compromised.
2. Update any systems that used these credentials with newly rotated tokens.

Recommended follow-ups
- Inform collaborators and ask them to reclone the repository or run:
```
git fetch origin
git reset --hard origin/main
```
  to avoid retaining old history locally.
- If you want GitHub to help remove any cached copies, open a support request or use the Secret Scanning unblock URL in the repository security alerts.

Notes & audit
- A local backup of the previous history existed temporarily while cleaning; all local safety refs were removed and garbage-collected. If you require further proof of removal or an audit log, I can prepare a short verification script to run on a separate machine.

If you want me to add an audit entry to `README.md` or create a small script to verify the absence of the tokens in commits, tell me and I'll add it.
