import { useCallback } from 'react'
import { useSimulationContext } from '../../context/SimulationContext'
import { useSimulation } from '../../hooks/useSimulation'
import './SimulationControls.css'

const SLIDERS = [
  { key: 'lambda',        label: 'Lambda λ',       min: -4,   max: 4,   step: 0.01, unit: 'σ',   color: '#00d4ff' },
  { key: 'rpm',           label: 'Speed / RPM',     min: -2,   max: 3,   step: 0.05, unit: 'σ',   color: '#00ff88' },
  { key: 'load',          label: 'Engine Load',     min: -2,   max: 3,   step: 0.05, unit: 'σ',   color: '#00ff88' },
  { key: 'ignitionAngle', label: 'Ignition Angle',  min: -4,   max: 4,   step: 0.05, unit: 'σ',   color: '#ffaa00' },
  { key: 'coBaseline',    label: 'CO Baseline',     min: -2,   max: 3,   step: 0.05, unit: 'σ',   color: '#ff3355' },
  { key: 'hcBaseline',    label: 'HC Baseline',     min: -2,   max: 3,   step: 0.05, unit: 'σ',   color: '#ff3355' },
]

function SliderRow({ config, value, onChange, disabled }) {
  const range = config.max - config.min
  const fill  = ((value - config.min) / range * 100).toFixed(1) + '%'

  return (
    <div className="slider-wrap">
      <div className="slider-header">
        <span className="slider-name">{config.label}</span>
        <span className="slider-value" style={{ color: config.color }}>
          {value.toFixed(2)}
          <span className="slider-unit">{config.unit}</span>
        </span>
      </div>
      <input
        type="range" min={config.min} max={config.max} step={config.step}
        value={value}
        onChange={e => onChange(config.key, parseFloat(e.target.value))}
        disabled={disabled}
        style={{ '--fill': fill, '--thumb-color': config.color }}
      />
      <div className="slider-limits">
        <span>{config.min}</span>
        <span>{config.max}</span>
      </div>
    </div>
  )
}

export default function SimulationControls() {
  const { state, setSlider, setFaultInject, setAutoCorrection } = useSimulationContext()
  const { start, stop } = useSimulation()

  const handleSlider = useCallback((key, val) => setSlider(key, val), [setSlider])

  return (
    <div className="controls-column panel panel-scan">
      <div className="panel-title">Control Panel</div>

      {/* Sliders */}
      <div className="sliders-section">
        {SLIDERS.map(cfg => (
          <SliderRow
            key={cfg.key}
            config={cfg}
            value={state.sliders[cfg.key]}
            onChange={handleSlider}
            disabled={false}
          />
        ))}
      </div>

      <div className="divider" />

      {/* Fault injection */}
      <div className="fault-row">
        <span className="slider-name">Fault Inject</span>
        <div className="fault-controls">
          <select
            value={state.faultInject ?? ''}
            onChange={e => setFaultInject(e.target.value || null)}
            disabled={state.isRunning && state.faultInject !== null}
          >
            <option value="">— None —</option>
            <option value="fault1">Fault 1 · Rich Mix</option>
            <option value="fault2">Fault 2 · Lean Mix</option>
            <option value="fault3">Fault 3 · Ignition</option>
          </select>
        </div>
      </div>

      <div className="divider" />

      {/* Auto-correction toggle */}
      <div className="toggle-wrap">
        <span className="toggle-label">Auto-Correction</span>
        <label className="toggle">
          <input
            type="checkbox"
            checked={state.autoCorrection}
            onChange={e => setAutoCorrection(e.target.checked)}
          />
          <span className="toggle-slider" />
        </label>
      </div>

      <div className="divider" />

      {/* Start / Stop button */}
      {!state.isRunning ? (
        <button
          className="btn btn-start"
          onClick={start}
          disabled={!state.backendReady}
        >
          ▶ START SIMULATION
        </button>
      ) : (
        <button className="btn btn-stop" onClick={stop}>
          ◼ STOP
        </button>
      )}

      {!state.backendReady && (
        <p className="controls-notice">Waiting for backend...</p>
      )}
    </div>
  )
}
