# Tier-2 Scenario Families (Proposal)

**Author's note.** These sixteen families were derived only from the stated
policy (per-person allowance, sharing permitted, multi-accounting prohibited)
and from ordinary intuitions about how people actually share devices,
networks, and payment methods, and how a rational abuser would try to stay
under a naive detector's radar. I deliberately weighted the set toward hard
negatives — legitimate or ambiguous clusters whose raw linkage signals
(shared device, shared network, shared payment token, synchronized timing)
look exactly like the textbook abuse signature — because a world made only of
clean examples doesn't exercise the boundary the policy actually cares about:
distinguishing "four people, one tablet" from "one person, four tablets." I
also included abuse families that do *not* look like a farm (slow burn, mule
recruitment, expiry-triggered rerolling) since an abuser who has seen a naive
detector once will stop looking like the first example. No family encodes a
demographic proxy; all behavior differences are functional (device
provisioning, work shift, verification posture, credit-claim cadence), not
identity-based.

## 1. Household Shared Device
- **Label:** legitimate
- **Clusters / size:** ~40 clusters, ~3–5 accounts each
- **Description:** A household (family, couple, roommates) shares one
  tablet or living-room PC. All accounts share the same `stable_device` and
  usually the same `network_group` (home wifi) at high reliability. Sessions
  cluster in evenings and weekends, sometimes with two members active within
  minutes of each other (parent hands the tablet to a kid). `claim_credits`
  happens near the start of each person's own cycle, offset by whenever each
  person originally signed up, so claims are not perfectly synchronized.
  Payment tokens, when present, are often shared (one adult's card) across 2+
  accounts.
- **Why it's hard:** This is the anchor case the policy explicitly protects,
  but the same device+network signal pair is the single strongest abuse
  fingerprint in the schema — a detector that fires on device/network
  co-location alone breaks the policy's core distinction on its most common
  legitimate instance.

## 2. Sequential Household Users
- **Label:** legitimate
- **Clusters / size:** ~30 clusters, ~3–4 accounts each
- **Description:** Same physical household device as family 1, but the
  members keep incompatible schedules (shift worker, student, retiree) so
  their sessions almost never overlap and instead tile across the day —
  account A mornings, account B afternoons, account C late night — every day
  of the week, for months. `stable_device` and `network_group` are shared at
  high reliability; `behavioral_template` differs per account (distinct
  endpoint-cost-tier mix and consumption pace per person).
- **Why it's hard:** Perfect temporal round-robining across accounts on one
  device is exactly the access pattern a rule like "accounts on this device
  are never active at the same time" would flag as one person avoiding
  concurrent-session detection, when it's actually just three people with
  disjoint schedules.

## 3. Shared Public Kiosk / Library Terminal
- **Label:** legitimate
- **Clusters / size:** ~15 clusters, ~8–20 accounts each
- **Description:** A public library, internet café, or community-center
  terminal used by many unrelated people over months. All accounts on a
  cluster share `stable_device` (the terminal) and `network_group` at high
  reliability, but `recovery_contact` and `payment_token` are always
  distinct or absent (most sessions are low-verification, browsing-only
  patrons). Signups trickle in over the whole observation window rather than
  bursting, concentrated in the terminal's opening hours (e.g. weekday
  daytime), with sparse, low-credit-tier usage per account.
- **Why it's hard:** High account count on one device/network pair, spread
  over a long window, is the classic farm shape by volume alone; the
  distinguishing signal is that verification/payment/recovery signals never
  co-occur across accounts and per-account usage is thin and one-off rather
  than sustained.

## 4. Small Business Turnover
- **Label:** legitimate
- **Clusters / size:** ~25 clusters, ~4–8 accounts each
- **Description:** A small studio or shop where staff use the product for
  work. Accounts are created in a slow, sequential trickle over the year as
  employees are hired (not a burst), all on the office `network_group` and
  often the company `payment_token` (enterprise or consumer card on file for
  reimbursement), weekday-business-hours usage only. Departed employees'
  accounts go quiet — `login` and `consume_credits` simply stop — rather than
  being reused or handed off.
- **Why it's hard:** Sequential account creation sharing one payment token
  and one network is often treated as the strongest fraud signal ("staged
  onboarding of sock puppets"); here it is ordinary hiring cadence, and the
  giveaway is that old accounts die out completely instead of being recycled
  or kept warm.

## 5. Account Recovery Migration
- **Label:** legitimate
- **Clusters / size:** ~35 clusters, 2 accounts each
- **Description:** A person loses access to their account (forgotten
  credentials, deleted email, phone swap) and creates a replacement. The old
  account's activity stops abruptly; the new one appears shortly after (same
  day to a few weeks later), often sharing `recovery_contact` or
  `payment_token` with the old one at medium/high reliability, sometimes on
  the same `stable_device`. The new account claims its own 25 credits at
  signup.
- **Why it's hard:** From the event tables this looks identical to "abandon
  account near quota exhaustion, spin up a fresh one for another free
  allocation" — the only real difference is a policy-level intent that isn't
  observable, so the case rests entirely on cadence (does the old account
  show pre-abandonment credit exhaustion and a gap, versus a live account
  kept in reserve) and on whether the pattern repeats more than once per
  cluster.

## 6. Roommate Turnover
- **Label:** ambiguous
- **Clusters / size:** ~25 clusters, ~3–6 accounts each
- **Description:** A shared apartment where the roommate roster changes
  every several months (typical of rented shared housing). The physical
  device/network stays constant (`stable_device`, `network_group` high
  reliability across the whole cluster lifetime), but each roommate brings
  their own `payment_token`, and accounts overlap in time for a while before
  one drops off and a new one appears at a similar cadence to family 5, except
  here it recurs multiple times in the same cluster as different roommates
  cycle through.
- **Why it's hard:** It's genuinely on the boundary between families 1/2 and
  a device-based farm: a naive detector sees one device supporting an
  unbounded, ever-changing stream of accounts across years and has no
  principled cutoff for "how many different people may ever share one
  tablet" — the policy permits it, but nothing stops an abuser from mimicking
  exactly this shape by periodically swapping the account attached to a
  device.

## 7. Classroom / Course Cohort
- **Label:** legitimate
- **Clusters / size:** ~10 clusters, ~15–30 accounts each
- **Description:** A school or bootcamp assigns the product for coursework.
  All accounts share `network_group` (campus wifi, high reliability) and an
  `education` email domain type, with `verification_level` clustered at
  `email` or `phone`. Usage is tightly synchronized to class sessions —
  everyone logs in and consumes credits within the same 90-minute windows on
  the same weekdays for a fixed term, then usage drops to near zero at the
  end of the semester for the whole cohort simultaneously.
- **Why it's hard:** Dozens of accounts firing `login`/`consume_credits`
  within the same narrow time windows on the same network is the single most
  bot-farm-looking temporal signature in the whole schema, yet it's produced
  by ordinary institutional scheduling; the semester-boundary drop-off
  (everyone stops at once, not just one) is the distinguishing feature.

## 8. Trial Credit Farm
- **Label:** abusive
- **Clusters / size:** ~10 clusters, ~5–15 accounts each
- **Description:** One operator scripts repeated signups using disposable
  email domains, claims the full 25 credits immediately (`claim_credits`
  right after `signup`, seconds apart), burns them fast on high-cost-tier
  generations, and abandons the account — repeating in bursts. Accounts in a
  cluster share `stable_device` and/or `network_group` at high reliability
  and often `behavioral_template` (identical timing-of-actions fingerprint
  since it's the same script). `verification_failed` events are common
  because the operator doesn't bother completing verification.
- **Why it's hard:** It's the textbook case, included as a calibration
  anchor rather than a genuine hard case — but note it must NOT be confused
  with family 3 (public kiosk), which shares device/network too; the
  discriminators are burst creation timing, disposable domains, and
  immediate full-allocation burn-and-abandon.

## 9. Serial Reroll on Expiry
- **Label:** abusive
- **Clusters / size:** ~12 clusters, ~3–6 accounts each
- **Description:** A single person exhausts their 25 credits early in each
  cycle, then — instead of waiting for renewal — creates a new account near
  the end of that cycle to claim a fresh allocation immediately, repeating
  every cycle or two. Accounts share `stable_device` and often
  `payment_token` or `recovery_contact` at high reliability, and each new
  account's `entitlement_cycles` row starts right as the previous account's
  `credits_consumed` hits its `credits_granted` ceiling. Verification level
  stays low (`none`/`email`) to make each signup fast.
- **Why it's hard:** It looks structurally identical to family 5 (recovery
  migration) — same device, same recovery signal, sequential accounts — and
  the *only* differences are (a) the old account isn't actually
  inaccessible, it's just quota-exhausted, and (b) the pattern repeats
  regularly on a cycle-aligned cadence rather than being a one-off; a
  detector keyed on "sequential account handoff on shared device" alone
  cannot tell this apart from legitimate recovery without the recurrence and
  quota-timing signal.

## 10. Credit Mule Ring
- **Label:** abusive
- **Clusters / size:** ~8 clusters, ~4–8 accounts each
- **Description:** An operator recruits several genuinely distinct people
  (friends, forum contacts, gig workers) to create their own accounts on
  their own devices and hand over the output/credits, often for a small
  payment. Each account has its own distinct `stable_device`,
  `network_group`, and `behavioral_template` — no device or network linkage
  at all — but they share one `payment_token` (the operator pays everyone
  from the same source, or reimburses via the same instrument) or a single
  `recovery_contact` the operator insists on holding for "account recovery
  purposes." Consumption on all accounts spikes in the same narrow windows
  when the operator is actively directing the batch of workers.
- **Why it's hard:** No device or network signal links these accounts at
  all — the only thread is a single shared payment or recovery signal, which
  is exactly the kind of weak, easily-missed linkage a device/network-based
  detector is not designed to weight heavily; each individual account, in
  isolation, is indistinguishable from an ordinary single legitimate user.

## 11. Device Reseller Fleet
- **Label:** ambiguous
- **Clusters / size:** ~20 clusters, ~2–4 accounts each
- **Description:** A refurbished-device reseller sells the same physical
  tablet/phone to a sequence of unrelated buyers over its lifetime. Each
  buyer creates their own account on it once; `stable_device` links all of
  them at high reliability, but there is total separation in time (weeks to
  months between owners, no overlap), and every other signal —
  `network_group`, `payment_token`, `recovery_contact`, `behavioral_template`
  — is completely distinct per account, because each owner uses the device
  in a different home/city.
- **Why it's hard:** `stable_device` is meant to be a strong identity signal,
  but hardware identifiers persist across ownership changes; a detector that
  treats "same device ID, multiple accounts, no overlap" as an the
  recovery-migration pattern (family 5) will misjudge it as one evasive
  person, when it is actually N genuinely unrelated people who happen to
  share a device history — this is a case where the *ground truth itself* is
  debatable even for a human reviewer without out-of-band information, hence
  ambiguous rather than a clean legitimate call.

## 12. Co-working Hot Desk
- **Label:** legitimate
- **Clusters / size:** ~20 clusters, ~5–10 accounts each
- **Description:** A co-working space's shared rental laptops or its office
  wifi is used by rotating freelancers/contractors, each with their own
  account. `network_group` (and sometimes `stable_device`, for the loaner
  laptops) links the cluster at high reliability; `payment_token` and
  `recovery_contact` are always distinct per person. Usage is business-hours
  weekday, with genuinely different consumption habits per member (some
  heavy daily users, some occasional), and occasional simultaneous overlap
  when two members work at adjacent desks at the same time.
- **Why it's hard:** It combines two "farm-like" ingredients at once — shared
  device *and* shared network, at volume — with the added twist that,
  unlike family 3's kiosk, usage is sustained and heavy per account (not
  one-off), which is normally the tell that separates "farm" from
  "legitimate shared terminal"; here sustained heavy use is legitimate
  because each account belongs to a different working professional.

## 13. Boarding-School / Long-Distance Family
- **Label:** ambiguous
- **Clusters / size:** ~20 clusters, ~2–3 accounts each
- **Description:** A parent pays for a child's account from the same family
  card used for their own account (or a sibling away at university), so
  `payment_token` links the cluster at high reliability — but the accounts
  are never on the same `stable_device` or `network_group` (different cities,
  different networks, non-overlapping time-zone rhythms), and their
  consumption habits, endpoint-cost-tier mix, and session timing look
  nothing alike.
- **Why it's hard:** This isolates the payment-token signal from every other
  corroborating signal: is a shared card, on its own, evidence of a single
  operator running multiple accounts, or just a family paying for more than
  one member the ordinary way the policy explicitly allows? The schema gives
  no way to distinguish "one wallet, one person" from "one wallet, several
  dependents" without weighing it against the device/network/behavioral
  divergence, which is exactly the judgment call worth testing.

## 14. Referral / Viral Signup Burst
- **Label:** legitimate
- **Clusters / size:** ~15 clusters, ~10–20 accounts each
- **Description:** An influencer or article drives a wave of genuinely
  independent strangers to sign up within the same day or two. Because many
  of them are in the same metro area or use the same major ISP, some end up
  sharing a coarse `network_group` observation at low/medium reliability
  purely by coincidence — but `stable_device`, `payment_token`, and
  `recovery_contact` are all distinct, and post-signup behavior (session
  timing, consumption pace) diverges immediately once the novelty wears off.
- **Why it's hard:** A sharp, synchronized spike of new accounts loosely
  co-located on network is superficially indistinguishable from a
  coordinated farm launch; the discriminator is that the *only* shared
  signal is weak/coarse network co-location with no device, payment, or
  behavioral corroboration, and the accounts' trajectories fan out
  immediately afterward instead of staying in lockstep.

## 15. Verification-Avoidant Privacy Cluster
- **Label:** ambiguous
- **Clusters / size:** ~15 clusters, ~4–8 accounts each
- **Description:** Privacy-conscious but genuinely distinct individuals who
  route through the same VPN exit node or privacy-focused mobile carrier,
  use disposable email domains by habit, and decline phone verification
  (`verification_level: none`, occasional `verification_failed` from typo'd
  attempts, not evasion). This produces a shared `network_group` at medium
  reliability across unrelated people, low verification uniformly, and
  otherwise ordinary, modest, individually-distinct consumption patterns.
- **Why it's hard:** Every individually-weak abuse signal in the schema
  (disposable domain, no verification, shared network, failed verification
  events) co-occurs in this cluster purely because privacy-seeking behavior
  correlates with all of them at once — it is the adversarial case for any
  scoring rule that sums risk signals rather than checking for actual
  cross-account behavioral or device coupling.

## 16. Slow-Burn Multi-Account Hoarder
- **Label:** abusive
- **Clusters / size:** ~8 clusters, ~3–5 accounts each
- **Description:** One person deliberately creates a new account every few
  months (not in a burst), each on a different device and network to avoid
  fingerprint overlap (spare phone, work laptop, a friend's tablet used
  once), and never uses more than a fraction of each account's allowance per
  cycle to stay under any consumption-based threshold. The only thread tying
  the cluster together is a shared `recovery_contact` (same backup email/
  phone set at signup on every account, because the person doesn't bother
  varying it) or a subtle shared `behavioral_template` (same idiosyncratic
  session rhythm and endpoint-tier preference across all of them, since it's
  the same human).
- **Why it's hard:** Nothing about the volume, timing, or intensity of any
  individual account looks anomalous — no burst, no exhaustion spike, no
  device/network reuse — so any detector keyed on speed, volume, or device
  fingerprinting will see five perfectly ordinary light users; the only
  route to the truth is the recovery-contact or fine-grained behavioral
  match, deliberately made as thin as possible.

## Summary

| # | Family | Label | Clusters |
|---|---|---|---|
| 1 | Household Shared Device | legitimate | ~40 |
| 2 | Sequential Household Users | legitimate | ~30 |
| 3 | Shared Public Kiosk / Library Terminal | legitimate | ~15 |
| 4 | Small Business Turnover | legitimate | ~25 |
| 5 | Account Recovery Migration | legitimate | ~35 |
| 6 | Roommate Turnover | ambiguous | ~25 |
| 7 | Classroom / Course Cohort | legitimate | ~10 |
| 8 | Trial Credit Farm | abusive | ~10 |
| 9 | Serial Reroll on Expiry | abusive | ~12 |
| 10 | Credit Mule Ring | abusive | ~8 |
| 11 | Device Reseller Fleet | ambiguous | ~20 |
| 12 | Co-working Hot Desk | legitimate | ~20 |
| 13 | Boarding-School / Long-Distance Family | ambiguous | ~20 |
| 14 | Referral / Viral Signup Burst | legitimate | ~15 |
| 15 | Verification-Avoidant Privacy Cluster | ambiguous | ~15 |
| 16 | Slow-Burn Multi-Account Hoarder | abusive | ~8 |

**Approximate totals:** ~308 clusters, ~1,725 accounts.
Label mix by cluster count: legitimate ~200 (65%), ambiguous ~80 (26%),
abusive ~38 (12%) — abusive share sits inside the requested 5–15% band.
