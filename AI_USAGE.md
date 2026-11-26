# AI usage

## Tools I used, and for what

I used ChatGPT to review my implementation, identify potential edge cases, explain possible weaknesses, and suggest additional test scenarios. I wrote the initial PII detection regex myself and used ChatGPT for code review and testing guidance rather than generating the initial regex. I also used Python, pytest, and manual test cases to validate the implementation.

## The PII cases my initial regex missed at first, and how I found them

My initial regex was written by me, not AI-generated. During testing and review, I checked Egyptian phone number formats that could be missed, including international prefixes, alternative separators, and different digit lengths. Examples included +201012345678, 00201012345678, +2 010 12345678, 002-010-1234-5678, and 0101234567. I used ChatGPT's review suggestions and manual counterexamples to identify potential gaps and improve coverage.

## One sentence on what my service does when the model is down, written for a pharmacist

If the prediction model is unavailable, the service automatically uses a rule-based fallback to provide an alternative shortage-risk estimate, so the pharmacist still receives a result while knowing it was not produced by the main prediction model.
