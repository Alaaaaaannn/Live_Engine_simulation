import { useSimulationContext } from '../../context/SimulationContext'
import './Panels.css'

function computeScore(lambdaCurrent, coCurrent, hcCurrent, noxCurrent, faultClass) {
  const lambdaPenalty = Math.abs(lambdaCurrent) * 20
  const emPenalty = Math.max(0, coCurrent) * 25
               + Math.max(0, hcCurrent) * 20
               + Math.max(0, noxCurrent) * 20
  const faultPenalty = faultClass !== 0 ? 10 : 0
  return Math.max(0, Math.min(100, 100 - lambdaPenalty - emPenalty - faultPenalty))
}

function scoreColor(s) {
  return s >= 80 ? 'var(--accent-green)'
       : s >= 55 ? 'var(--accent-cyan)'
       : s >= 30 ? 'var(--accent-amber)'
       :           'var(--accent-red)'
}
function scoreLabel(s) {
  return s >= 80 ? 'Optimal' : s >= 55 ? 'Degraded' : s >= 30 ? 'Warning' : 'Critical'
}

export default function HealthScorePanel() {
  const { state } = useSimulationContext()
  const { lambdaCurrent, coCurrent, hcCurrent, noxCurrent, currentFault } = state

  const score  = computeScore(lambdaCurrent, coCurrent, hcCurrent, noxCurrent, currentFault.class)
  const color  = scoreColor(score)
  const label  = scoreLabel(score)

  const radius        = 28
  const circumference = 2 * Math.PI * radius
  const fillLen       = (score / 100) * circumference

  const subs = [
    { name: 'Engine Efficiency',    value: Math.max(0, Math.min(100, 100 - Math.abs(lambdaCurrent) * 25)) },
    { name: 'Emission Compliance',  value: Math.max(0, Math.min(100, 100 - Math.max(0, coCurrent)*25 - Math.max(0, hcCurrent)*20 - Math.max(0, noxCurrent)*20)) },
    { name: 'Fuel Economy',         value: Math.max(0, Math.min(100, 100 - Math.max(0, Math.abs(lambdaCurrent)-0.3)*35)) },
    { name: 'Ignition Health',      value: currentFault.class === 3 ? Math.max(20, 95 - Math.abs(lambdaCurrent)*20) : 95 },
  ]

  return (
    <div className="panel health-score-panel">
      <div className="panel-title">Vehicle health score</div>
      <div className="health-body">

        {/* Compact ring */}
        <div className="health-ring-wrap">
          <svg width="70" height="70" viewBox="0 0 70 70">
            <circle cx="35" cy="35" r={radius} fill="none" stroke="var(--border-subtle)" strokeWidth="8" />
            <circle cx="35" cy="35" r={radius} fill="none" stroke={color} strokeWidth="8"
              strokeDasharray={`${fillLen} ${circumference}`}
              strokeLinecap="round"
              transform="rotate(-90 35 35)"
              style={{ transition: 'stroke-dasharray 0.8s ease, stroke 0.5s ease' }}
            />
            <text x="35" y="32" textAnchor="middle" dominantBaseline="middle"
              fill="var(--text-primary)" fontSize="18" fontWeight="600" fontFamily="var(--font-body)">
              {Math.round(score)}
            </text>
            <text x="35" y="48" textAnchor="middle" dominantBaseline="middle"
              fill={color} fontSize="8" fontWeight="500"
              fontFamily="var(--font-body)">{label}</text>
          </svg>
        </div>

        {/* Sub-scores */}
        <div className="health-subs">
          {subs.map(s => (
            <div key={s.name} className="health-sub-row">
              <span className="health-sub-name">{s.name}</span>
              <div className="health-sub-track">
                <div className="health-sub-fill" style={{ width: `${s.value}%`, background: scoreColor(s.value) }} />
              </div>
              <span className="health-sub-val mono" style={{ color: scoreColor(s.value) }}>
                {Math.round(s.value)}
              </span>
            </div>
          ))}
        </div>

      </div>
    </div>
  )
}
