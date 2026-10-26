# Initial requirements traceability

| Requirement | First software increment | Later validation |
|---|---|---|
| F-1 360° coverage | Sector and obstruction API/UI states | Full-circle walk test; ≥5 analyzed frames/sec/sector |
| F-2 density | Zone reading contract, density panel, synthetic scenarios, and a separate evaluation-only CSRNet benchmark candidate (not connected to dashboard) | Manual ground truth, day/night/rain accuracy; public benchmark does not validate persons/m² |
| F-3 count/fill | Headcount and capacity fields | Chokepoint tally and prediction |
| F-4 movement | API reserved for motion events | Convergence/counter-flow drills |
| F-5 zones/calibration | Authenticated setup API; saved zone polygons; experimental planar projective mapping estimates each zone's ground area from a four-corner rectangle with operator-entered width and depth | No camera lens/fisheye correction, camera-reposition detection, independent area accuracy checks, under-10-minute untrained-operator validation, or real-density checks yet; the mapping is not validated for 360° video |
| F-6 night | Confidence and health states | Night testing; visible degradation |
| F-7 face detection | Deferred; no face data in scaffold | Quality gate and measured recognition metrics |
| F-8 missing-person search | Deferred; case API design required | Top-five volunteer test and closure expiry |
| F-9 watchlist | Deferred; off by default | Written authorization, auto-expiry, demographic metrics |
| F-10 vehicle intrusion | Alert contract supports event | Vehicle entry trials |
| F-11 alert engine | Synthetic scenario controls drive threshold bands; alerts deduplicate for 60 seconds, can be acknowledged, and red alerts escalate after 30 seconds while the simulation runs | Threshold, deduplication, escalation and latency tests |
| F-12 loudspeaker link | No speaker release in scaffold | Approved multilingual templates; operator release only |
| F-13 operator dashboard | Initial operator view | Full setup, queue, health and sunlight/glove review |
| F-14 control-room dashboard | Deferred | Fleet map, assignment, trends and case queue |
| F-15 logging | Not yet persistent; API contract foundation | Immutable audit, clips, retention, search and exports |

This is a development scaffold, not an operational or validated safety system.
