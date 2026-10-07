# Review practice — five cases

Five small changes, each with one thing wrong. They are the cases printed in the
*Reviewing a Teammate's Pull Request* handout, except here they actually run.

## How to use these

Do them **alone** first. For each case write two lines:

1. What you would comment on the pull request.
2. The check that would prove you right.

Then run the file and see whether your check was the one that settles it:

```
python review-practice/case1_speedup.py
```

Each case prints the evidence and then, at the bottom, what the check shows. The
written answer key is at the back of the handout — reading it before you have
written your own two lines removes the exercise.

Case 5 has no code; it is a process case, in `case5_twins.md`.

## Why the check matters more than the answer

Spotting the bug is the part that does not transfer. The check does. "Run it on
data that cannot predict anything" catches look-ahead you have never seen before,
in code you did not write, including code an agent wrote while you were reading
something else.

Every case here is a defect class we can hit for real in this project. None of
them is anyone's work.

| File | Nickname | The defect |
|---|---|---|
| `case1_speedup.py` | The speed-up | Look-ahead: a refactor drops the `shift(1)` guard |
| `case2_best_of_24.py` | The best of 24 | The maximum of a sweep reported as if it were one test |
| `case3_green_test.py` | The green test | An assertion that is true for any implementation |
| `case4_quiet_fix.py` | The quiet fix | `fillna` invents data; a bare `except` hides the error |
| `case5_twins.md` | The twins | Duplicate pull requests, committed clutter, template body |

## After you have done them

Bring one line per case to the sync: what you would have said. Where your answer
and someone else's differ, that gap is the part worth the meeting time.
