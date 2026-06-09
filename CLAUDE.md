Standing rules for everything that follows in this session:

\- Never break MOCK\_MODE — the app must run end-to-end with no API key.

&#x20; Whenever you add a new Claude call, add a matching branch in mocks.py.

\- Keep it token-cheap: retrieval over stuffing, small inputs to the model.

\- Make minimal, focused changes per step. Don't refactor unrelated code.

\- After each step, make sure the smoke test still passes before stopping.

\- Preserve the existing UI aesthetic and the Mode 2 flow.

