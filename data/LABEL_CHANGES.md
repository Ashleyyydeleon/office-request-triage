# Label changes in eval_cases.csv

Every change to an expected label is recorded here, with the reason, so scores
from different prompt versions can be compared honestly. Labels were never
changed just because the AI disagreed; each change follows a routing decision.

## 2026-11-17/18: holdout set added (#64-#78)

15 new messages written by the project owner and a friend, framed as messages
sent to a company's IT team, after the prompt and policies were finished.
The project owner reviewed all 15 and set every category and urgency by hand,
before the tool was ever run on them. These cases must not be used to tune the
prompt or policies; any later fix should be checked on new cases instead.

The owner's reasoning on the judgment calls:

| Case | Message (short) | Label | Reason |
|---|---|---|---|
| #65 | Visitor causing chaos in the waiting area | Safety & Security / high | A possible security threat that could escalate. Visitors also fits, but the visitor policy says nothing about unruly visitors. |
| #66 | Lobby spill, no wet floor sign | Facilities / high | A small fix, but a slip could hurt someone and expose the company to liability. |
| #67 | Computer won't turn back on | IT & Equipment / high | The person is blocked on that exact device. A close call with medium since it's one laptop (the IT policy says medium); high because it could be someone critical, like an executive. |
| #68 | Wi-Fi down on the second floor | IT & Equipment / high | Connectivity for a whole floor. |
| #72 | Forgot my password | IT & Equipment / medium | IT can reset it or explain how. |
| #73 | Mailroom is locked | Deliveries / medium | If it's closed, carriers may leave packages at the door or take them back, delaying people expecting mail. |
| #75 | Elevator stuck, not sure if anyone is inside | Safety & Security or Facilities / high | Always treat it as if someone is inside. Both teams are involved, so either category counts as correct (written "Safety & Security\|Facilities"). |
| #77 | Lost visitor roaming the halls | Visitors / medium | No sign of a threat; it would become Safety & Security only with details like suspicious behavior or an unknown item. |

Requirement found while labeling: #67, #68 and #72 belong to an IT team, so a
new "IT & Equipment" category was added to prompt v5 (IT problems were part of
"Other" before). This was decided from the owner's labels, before the tool had
produced any answers on the holdout set. No earlier case was labeled as IT, so
no other labels changed.

Known disagreement: #67 is labeled high, while the IT policy (written earlier)
says one person's laptop is medium. If the AI follows the policy, this will show
up as a miss; it reflects a real gap between the written policy and the owner's
judgment, and was left as is rather than editing the policy to fit the label.

## 2026-11-16, with prompt v4: new "Safety & Security" category

Decision: medical emergencies, injuries and security threats involving people
get their own team. Building hazards stay with the team that owns the space.

| Case | Message (short) | Old | New |
|---|---|---|---|
| #28 | Someone slipped on the wet floor | Facilities | Safety & Security |
| #58 | Someone collapsed and isn't responding | Other | Safety & Security |
| #59 | Person without a badge asking for laptops | Other | Safety & Security |

Result: the AI got these 3 right, but also sent fires, gas, a chemical leak and
trapped people to Safety & Security, against the rule (category 92% -> 87%,
or 94% -> 81% with RAG).

## 2026-11-16, with prompt v5: every emergency goes to Safety & Security

Decision: one source of truth. Anything listed in policies/emergencies_and_safety.md
(fire or smoke, gas, flooding or water through ceilings, someone hurt or trapped,
security threats, hazardous chemicals) is routed to Safety & Security wherever it
happens; the team that owns the space follows up on repairs. This matches the
emergency policy's own rule: call security first. Urgency labels are unchanged.

| Case | Message (short) | Old | New |
|---|---|---|---|
| #22 | Water leaking from the ceiling | Facilities | Safety & Security |
| #25 | Strong smell of gas | Facilities | Safety & Security |
| #26 | Elevator stuck with someone inside | Facilities | Safety & Security |
| #35 | Microwave sparked, smoke coming out | Kitchen | Safety & Security |
| #51 | Fire alarm going off, smell smoke | Facilities | Safety & Security |
| #52 | Water pouring out, flooding hallway | Facilities | Safety & Security |
| #55 | Toaster on fire | Kitchen | Safety & Security |
| #56 | Delivery box leaking a chemical | Deliveries | Safety & Security |
| #57 | Guest locked inside the visitor lobby | Visitors | Safety & Security |

Not changed: high-urgency cases where people are blocked but not in danger
(#1, #3, #5, #8, #12, #16, #53, #54), and non-emergencies about safety topics
(#30 alarm testing, #60 fire drill, #62 first aid kit).

Caveat: 6 of these 9 new labels (#26, #35, #51, #55, #56, #57) match what the
AI already answered in v4 without RAG (7 with RAG, adding #25). So part of any
v5 gain comes from the labels moving toward the AI, even though the rule came
from the emergency policy. Read the before/after numbers with that in mind;
the holdout set (written later, by other people) is the fair test.
