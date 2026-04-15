import { animated, useTrail } from '@react-spring/web'
import { useSimulationContext } from '../../context/SimulationContext'
import './Panels.css'

export default function ShapPanel() {
  const { state } = useSimulationContext()
  const features = state.shapFeatures ?? []

  // Animate each bar from 0 → actual importance (as a 0-100 number)
  const trail = useTrail(features.length, {
    from: { progress: 0, opacity: 0 },
    to:   { progress: 1, opacity: 1 },
    reset: true,
    config: { mass: 1, tension: 180, friction: 22 },
  })

  return (
    <div className="panel shap-panel">
      <div className="panel-title">SHAP Feature Importance</div>

      {features.length === 0 ? (
        <div className="dt-empty mono">Available on fault detection</div>
      ) : (
        <div className="shap-bars">
          {features.map((f, i) => {
            const targetPct = (f.importance * 100).toFixed(1)
            return (
              <div key={f.feature} className="shap-row">
                <span className="shap-feature mono">{f.feature}</span>
                <div className="shap-bar-track">
                  <animated.div
                    className="shap-bar-fill"
                    style={{
                      opacity: trail[i].opacity,
                      width: trail[i].progress.to(p => `${p * f.importance * 100}%`),
                      background: i < 3
                        ? 'linear-gradient(90deg, #00d4ff, #00ff88)'
                        : 'linear-gradient(90deg, #1a3a5c, #2a5a8c)',
                    }}
                  />
                </div>
                <span className="shap-val mono">{targetPct}%</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
