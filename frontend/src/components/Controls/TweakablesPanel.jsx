import { useEffect, useState, useCallback } from 'react'
import { getRuntimeConfig, postRuntimeConfig } from '../../api/client'
import { useSimulationContext } from '../../context/SimulationContext'
import './TweakablesPanel.css'

const FAULT_LABELS = {
  fault1: 'Rich  (λ –)',
  fault2: 'Lean  (λ +)',
  fault3: 'Ignition  (θ)',
}
const CLASS_LABELS = {
  '0': 'Normal',
  '1': 'Rich Mixture',
  '2': 'Lean Mixture',
  '3': 'Ignition Fault',
}

export default function TweakablesPanel() {
  const { dispatch } = useSimulationContext()
  const [open, setOpen]         = useState(false)
  const [section, setSection]   = useState('thresholds')
  const [cfg, setCfg]           = useState(null)
  const [defaults, setDefaults] = useState(null)
  const [busy, setBusy]         = useState(false)
  const [err, setErr]           = useState(null)

  const loadConfig = useCallback(async () => {
    try {
      setBusy(true)
      const c = await getRuntimeConfig()
      setCfg(c)
      if (defaults === null) setDefaults(JSON.parse(JSON.stringify(c)))
      dispatch({ type: 'SET_RUNTIME_CONFIG', payload: c })
      setErr(null)
    } catch (e) {
      setErr(e?.message || 'Failed to load runtime config')
    } finally {
      setBusy(false)
    }
  }, [defaults, dispatch])

  useEffect(() => { loadConfig() }, [loadConfig])

  const apply = async () => {
    if (!cfg) return
    try {
      setBusy(true)
      const r = await postRuntimeConfig({
        thresholds:      cfg.thresholds,
        fault_offsets:   cfg.fault_offsets,
        ctrl_step_fuel:  cfg.ctrl_step_fuel,
        ctrl_step_spark: cfg.ctrl_step_spark,
      })
      setCfg(r)
      dispatch({ type: 'SET_RUNTIME_CONFIG', payload: r })
      setErr(null)
    } catch (e) {
      setErr(e?.message || 'Apply failed')
    } finally {
      setBusy(false)
    }
  }

  const reset = async () => {
    try {
      setBusy(true)
      const r = await postRuntimeConfig({ reset: true })
      setCfg(r)
      setDefaults(JSON.parse(JSON.stringify(r)))
      dispatch({ type: 'SET_RUNTIME_CONFIG', payload: r })
      setErr(null)
    } catch (e) {
      setErr(e?.message || 'Reset failed')
    } finally {
      setBusy(false)
    }
  }

  if (!cfg) {
    return (
      <div className="tweakables-panel panel">
        <button className="tweakables-toggle" onClick={() => setOpen(o => !o)}>
          TWEAKABLES <span className="tweakables-state">{busy ? '...' : err ? 'OFFLINE' : 'READY'}</span>
        </button>
      </div>
    )
  }

  const setThr = (cls, v) =>
    setCfg(c => ({ ...c, thresholds: { ...c.thresholds, [cls]: v } }))

  const setOffset = (fkey, v) => setCfg(c => ({
    ...c,
    fault_offsets: {
      ...c.fault_offsets,
      [fkey]: { ...c.fault_offsets[fkey], delta: v },
    },
  }))

  return (
    <div className="tweakables-panel panel">
      <button className="tweakables-toggle" onClick={() => setOpen(o => !o)}>
        <span>TWEAKABLES</span>
        <span className="tweakables-state">{open ? '−' : '+'}</span>
      </button>

      {open && (
        <div className="tweakables-body">
          {err && <div className="tweakables-error">{err}</div>}

          <div className="tweakables-tabs">
            {['thresholds', 'faults', 'controller'].map(t => (
              <button
                key={t}
                className={`tweakables-tab ${section === t ? 'on' : ''}`}
                onClick={() => setSection(t)}
              >
                {t.toUpperCase()}
              </button>
            ))}
          </div>

          {section === 'thresholds' && (
            <div className="tweakables-section">
              <p className="tweakables-hint">
                Confidence threshold — predictions below this are gated to Normal.
              </p>
              {['0', '1', '2', '3'].map(cls => (
                <div key={cls} className="tweakables-row">
                  <label className="tweakables-label">{CLASS_LABELS[cls]}</label>
                  <input
                    type="range" min={0} max={1} step={0.01}
                    value={cfg.thresholds[cls] ?? 0.5}
                    onChange={(e) => setThr(cls, Number(e.target.value))}
                  />
                  <span className="tweakables-value">{(cfg.thresholds[cls] ?? 0).toFixed(2)}</span>
                </div>
              ))}
            </div>
          )}

          {section === 'faults' && (
            <div className="tweakables-section">
              <p className="tweakables-hint">
                Fault injection magnitudes (standardised units).
              </p>
              {Object.entries(FAULT_LABELS).map(([fkey, label]) => (
                <div key={fkey} className="tweakables-row">
                  <label className="tweakables-label">{label}</label>
                  <input
                    className="tweakables-number"
                    type="number" step={0.1}
                    value={cfg.fault_offsets?.[fkey]?.delta ?? 0}
                    onChange={(e) => setOffset(fkey, Number(e.target.value))}
                  />
                </div>
              ))}
            </div>
          )}

          {section === 'controller' && (
            <div className="tweakables-section">
              <p className="tweakables-hint">
                Per-cycle correction step sizes.  Smaller = slower, smoother heal.
              </p>
              <div className="tweakables-row">
                <label className="tweakables-label">Fuel trim</label>
                <input
                  type="range" min={0.005} max={0.20} step={0.005}
                  value={cfg.ctrl_step_fuel}
                  onChange={(e) => setCfg(c => ({ ...c, ctrl_step_fuel: Number(e.target.value) }))}
                />
                <span className="tweakables-value">{cfg.ctrl_step_fuel.toFixed(3)}</span>
              </div>
              <div className="tweakables-row">
                <label className="tweakables-label">Spark adv</label>
                <input
                  type="range" min={0.01} max={0.30} step={0.005}
                  value={cfg.ctrl_step_spark}
                  onChange={(e) => setCfg(c => ({ ...c, ctrl_step_spark: Number(e.target.value) }))}
                />
                <span className="tweakables-value">{cfg.ctrl_step_spark.toFixed(3)}</span>
              </div>
            </div>
          )}

          <div className="tweakables-actions">
            <button className="tweakables-btn primary" onClick={apply} disabled={busy}>APPLY</button>
            <button className="tweakables-btn"          onClick={reset} disabled={busy}>RESET</button>
          </div>
        </div>
      )}
    </div>
  )
}
