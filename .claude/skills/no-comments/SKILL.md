---
name: no-comments
description: Remove the comments your branch adds that REVIEW.md "Comments" doesn't keep, and make the code explain itself instead. Use before opening a PR or after writing code, alongside review-pr.
---

# No comments

Input: files or a diff, or nothing for the current branch against its base
(default `main`), working tree included. Only change your own branch.

1. **Find.** Launch one `comment-reviewer` agent on the scope.
2. **Check its report.** Reject:
   - any flag outside the scope;
   - a `DROP` of a comment that meets REVIEW.md's keep list;
   - a `KEEP` whose reason doesn't hold. Check the claim in the code; if it's
     still unclear, drop the comment.

   If you reject most of the report, rerun the agent once and say what it got
   wrong. If the second report fails too, stop and report it.
3. **Drop** every accepted `DROP`. Move history worth keeping to the PR
   description.
4. **Reshape.** For each `RESHAPE`, make the smallest in-scope change: a
   rename, a small extracted function, or a test that pins the behaviour.
   Then delete the comment. If the fix is outside the scope, delete the
   comment anyway and list the fix as open. Never add a guard just to justify
   a comment.
5. **Verify.** `pytest -q -m "not network"` and
   `ruff check . && ruff format .`. A comment-only change can't fail them; a
   reshape can.
6. **Report** the drops, reshapes, comments kept with their reason, and open
   items.
