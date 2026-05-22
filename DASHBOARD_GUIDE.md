# Dashboard Guide — Parameters, Faults, and Panels

A reference for everything on screen in the Engine Fault Simulator: every slider, every fault injection, every chart, every status panel. Use this when explaining the demo to someone new, or when you forget what a particular readout actually means.

---

## 1. Parameters (the six sliders in **Controls**)

All six are *engine state* inputs the model treats as the current operating point. The slider value is a z‑score; the readout shows physical units (from `frontend/src/utils/units.js`). "Threshold" below means the value where the engine starts behaving badly in real life, not a UI clamp.

| Slider | Physical meaning | Nominal | Why it shouldn't drift far |
|---|---|---|---|
| **Lambda λ** | Air‑fuel ratio normalized to stoichiometric. λ = 1.0 → exactly 14.7 g air per 1 g petrol. Below 1 = rich (excess fuel), above 1 = lean (excess air). | 1.00 (target ±0.02) | Catalyst only works inside ±~5 % of λ = 1. <1 → unburnt CO/HC poison the cat; >1 → combustion runs hot, NOx spikes, knock risk. |
| **Speed / RPM** | Crankshaft revolutions per minute. | 2500 rpm | Too low → stall, lean misfire. Too high (>~6000 on this class of engine) → mechanical stress, oil starvation, valve float. |
| **Engine Load** | % of max torque the engine is being asked to produce (throttle/load demand). | 50 % | Sustained 100 % load with off‑target λ multiplies thermal stress on pistons, exhaust valves and the cat by the same factor the fuel flow does. |
| **Ignition Angle** | Spark timing in degrees Before Top Dead Center. Tells the ECU how early to fire the plug before the piston reaches the top of the bore. | 15° BTDC | Too advanced (>~25°) → spark fires before the mixture is ready → knock / detonation, eats pistons and rings. Too retarded (<~5°) → late combustion → exhaust temps climb past 900 °C → catalyst meltdown. |
| **CO Baseline** | Carbon monoxide in the raw exhaust, % by volume. Direct rich‑mixture marker. | ~0.5 % vol | >~1 % sustained = catalyst overheats trying to oxidize it, and CO is itself toxic / regulated (Euro‑6 limit ≈ 1.0 g/km). |
| **HC Baseline** | Unburnt hydrocarbons in raw exhaust, ppm. Marker for misfire or incomplete combustion. | ~200 ppm | >~500 ppm sustained = raw fuel reaching the cat, which combusts it internally and can melt the substrate (>900 °C). |

All sliders show σ as a secondary readout because the model still consumes z‑scores; the physical number is the human‑readable view of the same input.

---

## 2. The three fault injections

These come from `backend/config.py` (`DEFAULT_FAULT_OFFSETS`). Selecting one and clicking **Inject** adds the offset to the live sensor window before it hits the model.

| Fault | What it perturbs | Default offset | What it represents in a real car |
|---|---|---|---|
| **Fault 1 — Rich Mixture (λ−)** | Lambda channel, **−1.5σ** (≈ λ 0.93). Also bumps CO +1.2σ, HC +0.7σ, NOx −0.6σ. | −1.5σ on λ | Injector stuck slightly open, leaky fuel pressure regulator, faulty MAF undermeasuring airflow. Engine gets too much fuel. CO and HC climb (incomplete combustion), NOx falls (cooler flame). |
| **Fault 2 — Lean Mixture (λ+)** | Lambda channel, **+1.5σ** (≈ λ 1.07). Also bumps CO −0.5σ, HC +0.3σ, NOx +1.1σ. | +1.5σ on λ | Vacuum leak, clogged injector, failing fuel pump, exhaust leak fooling the O₂ sensor. Engine gets too much air. CO falls, but NOx jumps because the flame is hotter and oxygen‑rich. Knock and piston damage if sustained. |
| **Fault 3 — Ignition Fault (θ)** | IgnitionAngle channel, **+2.0σ** (≈ +10° too advanced). Bumps CO +0.4σ, HC +1.0σ, NOx +0.5σ. | +2.0σ on θ | Worn spark plug, weak coil pack, ECU timing map corruption, knock sensor failure. Causes misfires (so the giant HC spike — raw fuel goes out the exhaust port and into the cat). The 3D engine shows this as irregular "jolt" shake. |

The same numbers can be tuned live in **Tweakables → FAULTS**, which is why those fields allow negative values — the sign encodes direction.

---

## 3. Dashboards / panels — what each is actually measuring

### LIVE ENGINE (3D viewer + status strip)
A real‑time visualization of the engine. Vibrates harder when faults are active. Spark plugs / coil glow red on ignition faults; intake manifold / fuel rail / injectors glow red on rich or lean faults. The bottom status strip shows the current FAULT, λ, CO, NOx — the four numbers a human first looks at.

### Controls (sliders + fault inject + auto‑correction + Start/Stop)
Lets you steer the engine state and inject faults. "Auto‑correction" toggles the ECU control loop — off = engine drifts free, on = the digital twin proposes a fuel‑trim / spark correction every cycle.

### Tweakables (collapsible)
Power‑user knobs: classifier *confidence thresholds* per class (predictions below threshold fall back to "Normal"), per‑fault *injection magnitudes*, and ECU *step sizes* for fuel and spark correction.

### Fault status panel
- **Fault name + Confidence**: softmax output of the BiLSTM classifier on the last 30‑sample window.
- **Lambda**: current λ, physical + σ.
- **Parameter state**: rule‑based readout of which slider is most out‑of‑band, with a 0–100 % severity.
- **Self‑healing bar**: how far λ has been pulled back toward 1.0 since the fault appeared (1.5σ = 0 %, 0.05σ = 100 %).

### Temporal stability panel
Looks at the last few classifier outputs and reports the majority label + an agreement %. Low agreement = the classifier is flipping between classes (sensor noise, transient). High agreement = stable diagnosis. Color: red < 60 %, amber < 85 %, green ≥ 85 %.

### Vehicle health score
Single 0–100 number combining lambda deviation + CO/HC/NOx penalties + active‑fault penalty. Plus four sub‑scores: Engine Efficiency, Emission Compliance, Fuel Economy, Ignition Health. Optimal ≥ 80, Degraded ≥ 55, Warning ≥ 30, Critical below.

### SHAP feature importance
When a fault is detected, the backend runs SHAP on the BiLSTM and returns per‑feature importance. Top 3 bars are cyan — these are the sensor channels the model most relied on to make the diagnosis ("why" behind the prediction).

### ECU action log (Digital Twin Log)
One row per cycle: the proposed control action (fuel trim ± σ, spark advance ± σ), the twin's predicted next‑cycle λ, and whether the action was **Approved** or **Rejected**. Rejection happens when the twin predicts the action would push λ further from target.

### Lambda Convergence chart
λ over time (cycles, ~0.5 s each).
- **Blue solid** — actual λ from the sensor window.
- **Purple dashed** — twin's one‑step‑ahead prediction.
- **Green band** — stoichiometric target zone (0.98–1.02).

When healing works, both lines fall back into the green band.

### Emission Levels chart
CO (red), HC (amber), NOx (purple) over cycles. Three different physical units, so the y‑axis stays in σ for relative comparison; the tooltip shows the per‑series physical reading.

### Environmental impact counter
Accumulates *while a fault is active and being corrected*: CO/HC/NOx mass that would have been emitted without correction, fuel waste avoided (rich faults only), and catalyst over‑temp seconds prevented. Resets on each new Start.

### "Without AI twin" panel
Shows up while a fault is active and not yet healed. Compares detection time (AI twin 0.5 s vs OBD‑II ~20–60 s vs mechanic "days") and lists the consequences a real ECU/driver would face if this fault went uncorrected — Euro‑6 multiples, catalyst meltdown thresholds, etc.

### Healing report (comparison panel)
Appears after a fault has been detected *and* λ has returned to ±0.05σ. Side‑by‑side **Fault → Healed** table for λ, CO, HC, NOx in physical units, with % change. Also reports cycles taken and wall‑clock seconds to heal.

### Previous simulations / history
Sessions are persisted server‑side; this is a list of past runs you can open to replay their metrics.

---

## One mental model that ties it all together

λ is the master signal. Faults 1 & 2 push λ off target directly; Fault 3 pushes ignition timing off target, which then drives λ off through misfires. The classifier reads the 30‑step sensor window and labels the fault; the twin predicts what λ will do next under any proposed fuel / spark correction; the ECU only applies corrections the twin endorses. Every panel on screen is either measuring **how far off λ (and emissions) currently are**, **what the model thinks is wrong**, **what action it proposes**, or **how much damage was avoided by acting fast**.
