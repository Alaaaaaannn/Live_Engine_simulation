import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { useSimulationContext } from '../../context/SimulationContext'
import { useSimulation } from '../../hooks/useSimulation'
import { useHashRoute } from '../../hooks/useHashRoute'
import { useAuth } from '../../context/AuthContext'
import './Header.css'

export default function Header() {
  const { state } = useSimulationContext()
  const { switchEngine } = useSimulation()
  const { path, navigate } = useHashRoute()
  const { user, logout } = useAuth()
  const titleRef  = useRef(null)
  const statusRef = useRef(null)

  const onDashboard = path === '/' || path === '' || path === '/dashboard'
  const onHistory   = path.startsWith('/history')

  // Glitch-type-in animation on mount
  useEffect(() => {
    const el = titleRef.current
    if (!el) return
    gsap.from(el, {
      x: -20, duration: 0.8,
      ease: 'power3.out', delay: 0.3, clearProps: 'transform',
    })
  }, [])

  return (
    <header className="header">
      <div className="header-left">
        <span className="header-mark" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 17l6-6 4 4 8-8" />
            <path d="M14 7h7v7" />
          </svg>
        </span>
        <h1 className="header-title" ref={titleRef}>
          Digital Twin
          <span className="header-subtitle"> · Engine Fault Simulator</span>
        </h1>

        <nav className="header-nav" aria-label="Primary">
          <button
            type="button"
            className={`header-nav-tab ${onDashboard ? 'active' : ''}`}
            onClick={() => navigate('/')}
          >
            Dashboard
          </button>
          <button
            type="button"
            className={`header-nav-tab ${onHistory ? 'active' : ''}`}
            onClick={() => navigate('/history')}
          >
            Previous simulations
          </button>
        </nav>
      </div>

      <div className="header-right">
        <label className="header-engine-label">Engine</label>
        <select
          value={state.engineId}
          onChange={e => switchEngine(e.target.value)}
          disabled={state.isRunning}
        >
          <option value="gengine1">gengine1</option>
          <option value="gengine2">gengine2</option>
          <option value="pengines">pengines</option>
        </select>

        {state.cycleNumber > 0 && (
          <span className="header-cycle mono">Cycle <span className="text-cyan">{state.cycleNumber}</span></span>
        )}

        {user && (
          <div className="header-user">
            <span className="header-user-email" title={user.email}>{user.email}</span>
            <button type="button" className="header-logout" onClick={logout}>
              Log out
            </button>
          </div>
        )}
      </div>
    </header>
  )
}
