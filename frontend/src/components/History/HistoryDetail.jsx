import { useEffect, useState } from 'react'
import { get, subscribe } from '../../history/store'
import LambdaChart    from '../Charts/LambdaChart'
import EmissionsChart from '../Charts/EmissionsChart'
import { formatPhysical, formatBoth } from '../../utils/units'
import './History.css'

function formatTs(ts) {
  try { return new Date(ts).toLocaleString() } catch { return '' }
}
function fmt(n, d = 3) {
  return (typeof n === 'number' && Number.isFinite(n)) ? n.toFixed(d) : '—'
}
// Physical-unit readout for any stored z-score; "—" when the value is missing
// so existing rows still render the same dash instead of "λ NaN".
function fmtPhys(channel, n) {
  return (typeof n === 'number' && Number.isFinite(n)) ? formatPhysical(channel, n) : '—'
}
function fmtBoth(channel, n) {
  return (typeof n === 'number' && Number.isFinite(n)) ? formatBoth(channel, n) : '—'
}
function faultBadgeClass(fc) {
  if (fc == null || fc === 0) return 'badge-normal'
  if (fc === 3)               return 'badge-critical'
  return 'badge-fault'
}

function Section({ title, children, count }) {
  return (
    <section className="hist-section panel">
      <div className="panel-title">
        {title} {count != null && <span className="hist-section-count">· {count}</span>}
      </div>
      {children}
    </section>
  )
}

function KVList({ items }) {
  return (
    <div className="hist-kv-list">
      {items.map(([k, v]) => (
        <div className="hist-kv-row" key={k}>
          <span className="hist-kv-k">{k}</span>
          <span className="hist-kv-v mono">{v}</span>
        </div>
      ))}
    </div>
  )
}

export default function HistoryDetail({ id, onBack, onRename, onDelete }) {
  const [run, setRun] = useState(() => get(id))
  useEffect(() => {
    setRun(get(id))
    const off = subscribe(() => setRun(get(id)))
    return off
  }, [id])

  if (!run) {
    return (
      <div className="hist-page">
        <header className="hist-page-header">
          <button className="hist-back-btn" onClick={onBack}>Back</button>
          <h2 className="hist-page-title">Simulation not found</h2>
        </header>
        <div className="hist-empty panel">
          <p>This run was deleted or never existed.</p>
        </div>
      </div>
    )
  }

  const o   = run.finalObservations  || {}
  const fs  = run.faultSnapshot
  const hs  = run.healedSnapshot
  const imp = run.impactStats        || {}
  const rc  = run.runtimeConfig

  const params = [
    ['Engine',          run.engineId],
    ['Session ID',      run.sessionId],
    ['Started',         formatTs(run.startedAt)],
    ['Recorded',        formatTs(run.createdAt)],
    ['Cycles',          run.cycleCount],
    ['Auto-correction', run.autoCorrection ? 'On' : 'Off'],
  ]
  const sliders = [
    ['Lambda λ',       fmtBoth('lambda',        run.sliders?.lambda)],
    ['Speed / RPM',    fmtBoth('rpm',           run.sliders?.rpm)],
    ['Engine load',    fmtBoth('load',          run.sliders?.load)],
    ['Ignition angle', fmtBoth('ignitionAngle', run.sliders?.ignitionAngle)],
    ['CO baseline',    fmtBoth('coBaseline',    run.sliders?.coBaseline)],
    ['HC baseline',    fmtBoth('hcBaseline',    run.sliders?.hcBaseline)],
  ]
  const observations = [
    ['λ current',      fmtBoth('lambda', o.lambdaCurrent)],
    ['λ predicted',    fmtBoth('lambda', o.lambdaPredicted)],
    ['CO',             fmtPhys('co',  o.coCurrent)],
    ['HC',             fmtPhys('hc',  o.hcCurrent)],
    ['NOx',            fmtPhys('nox', o.noxCurrent)],
    ['Last fuel trim', `${fmt(o.lastFuelTrim, 4)} σ`],
    ['Last spark adv', `${fmt(o.lastSparkAdv, 4)} σ`],
    ['Converged',      o.converged ? 'Yes' : 'No'],
    ['Stability label',     o.stabilityLabel ?? '—'],
    ['Stability agreement', typeof o.stabilityAgreement === 'number' ? `${(o.stabilityAgreement * 100).toFixed(0)}%` : '—'],
  ]

  return (
    <div className="hist-page hist-detail">
      <header className="hist-page-header">
        <button className="hist-back-btn" onClick={onBack}>Back</button>
        <div className="hist-detail-titlewrap">
          <h2 className="hist-page-title">{run.name}</h2>
          <div className="hist-card-meta">
            <span>{formatTs(run.createdAt)}</span>
            <span className="hist-sep">·</span>
            <span className="mono">{run.engineId}</span>
            <span className="hist-sep">·</span>
            <span>{run.cycleCount} cycles</span>
            <span className={`badge ${faultBadgeClass(run.finalFault?.class)}`}>
              Final: {run.finalFault?.name ?? 'Normal'}
            </span>
          </div>
        </div>
        <div className="hist-card-actions">
          <button className="hist-icon-btn" onClick={() => onRename(run.id, run.name)}>Rename</button>
          <button className="hist-icon-btn danger" onClick={() => onDelete(run.id, run.name)}>Delete</button>
        </div>
      </header>

      <div className="hist-detail-grid">
        <Section title="Run parameters"><KVList items={params} /></Section>
        <Section title="Initial slider values"><KVList items={sliders} /></Section>
        <Section title="Final observations"><KVList items={observations} /></Section>

        <Section title="Final classification">
          <KVList items={[
            ['Gated class', run.finalFault?.class ?? 0],
            ['Name',        run.finalFault?.name  ?? 'Normal'],
            ['Confidence',  typeof run.finalFault?.confidence === 'number' ? `${(run.finalFault.confidence * 100).toFixed(2)}%` : '—'],
            ['Raw class',   run.finalFault?.raw   ?? 0],
          ]} />
        </Section>

        {fs && (
          <Section title="Fault snapshot (at detection)">
            <KVList items={[
              ['Cycle',       fs.cycle],
              ['Fault name',  fs.faultName],
              ['λ',           fmtBoth('lambda', fs.lambda)],
              ['CO',          fmtPhys('co',  fs.co)],
              ['HC',          fmtPhys('hc',  fs.hc)],
              ['NOx',         fmtPhys('nox', fs.nox)],
            ]} />
          </Section>
        )}
        {hs && (
          <Section title="Healed snapshot">
            <KVList items={[
              ['Cycle',       hs.cycle],
              ['λ',           fmtBoth('lambda', hs.lambda)],
              ['CO',          fmtPhys('co',  hs.co)],
              ['HC',          fmtPhys('hc',  hs.hc)],
              ['NOx',         fmtPhys('nox', hs.nox)],
            ]} />
          </Section>
        )}

        <Section title="Environmental impact (estimated)">
          <KVList items={[
            ['CO saved',          `${fmt(imp.coSavedG, 2)} g`],
            ['HC saved',          `${fmt(imp.hcSavedMg, 1)} mg`],
            ['NOx saved',         `${fmt(imp.noxSavedMg, 1)} mg`],
            ['Fuel saved',        `${fmt(imp.fuelSavedMl, 2)} ml`],
            ['Catalyst protected',`${fmt(imp.catalystProtectedS, 1)} s`],
          ]} />
        </Section>

        {Array.isArray(run.shapFeatures) && run.shapFeatures.length > 0 && (
          <Section title="Top SHAP features" count={run.shapFeatures.length}>
            <ul className="hist-shap-list">
              {run.shapFeatures.map((f, i) => (
                <li key={i} className="hist-shap-row">
                  <span>{f.feature}</span>
                  <span className="mono">{(f.importance * 100).toFixed(2)}%</span>
                  <span className="hist-shap-bar">
                    <span style={{ width: `${Math.min(100, Math.abs(f.importance) * 100)}%` }} />
                  </span>
                </li>
              ))}
            </ul>
          </Section>
        )}

        {rc && (
          <Section title="Runtime config snapshot">
            <KVList items={[
              ['Fuel step',    fmt(rc.ctrl_step_fuel, 3)],
              ['Spark step',   fmt(rc.ctrl_step_spark, 3)],
              ['Thresholds',   JSON.stringify(rc.thresholds || {})],
              ['Fault offsets', JSON.stringify(rc.fault_offsets || {})],
            ]} />
          </Section>
        )}
      </div>

      <div className="hist-charts-row">
        <div className="hist-chart-wrap">
          <LambdaChart    data={run.lambdaHistory    || []} height={360} title="λ convergence (history)" />
        </div>
        <div className="hist-chart-wrap">
          <EmissionsChart data={run.emissionsHistory || []} height={360} title="Emissions (history)"     />
        </div>
      </div>

      {Array.isArray(run.twinLog) && run.twinLog.length > 0 && (
        <Section title="Digital twin log" count={run.twinLog.length}>
          <div className="hist-twinlog">
            <div className="hist-twinlog-head">
              <span>Cycle</span><span>Fuel trim (σ)</span><span>Spark adv (σ)</span><span>λ pred</span><span>Approved</span><span>Fault</span>
            </div>
            {run.twinLog.map((e, i) => (
              <div className="hist-twinlog-row mono" key={i}>
                <span>{e.cycle}</span>
                <span>{fmt(e.fuelTrim, 3)}</span>
                <span>{fmt(e.sparkAdv, 3)}</span>
                <span>{fmtPhys('lambda', e.lambdaPred)}</span>
                <span className={e.approved ? 'text-green' : 'text-red'}>{e.approved ? 'Yes' : 'No'}</span>
                <span>{e.faultName}</span>
              </div>
            ))}
          </div>
        </Section>
      )}
    </div>
  )
}
