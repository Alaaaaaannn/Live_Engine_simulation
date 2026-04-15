import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { useSimulationContext } from '../../context/SimulationContext'
import { useSimulation } from '../../hooks/useSimulation'
import './Header.css'

export default function Header() {
  const { state } = useSimulationContext()
  const { switchEngine } = useSimulation()
  const titleRef  = useRef(null)
  const statusRef = useRef(null)

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
        <span className="header-diamond">◆</span>
        <h1 className="header-title" ref={titleRef}>
          AI DIGITAL TWIN
          <span className="header-subtitle"> · ENGINE FAULT SIMULATOR</span>
        </h1>
      </div>

      <div className="header-center">
        <span ref={statusRef} className={`header-status ${state.backendReady ? 'ready' : 'connecting'}`}>
          <span className="status-dot" />
          {state.backendReady ? 'SYSTEMS ONLINE' : 'CONNECTING...'}
        </span>
      </div>

      <div className="header-right">
        <label className="header-engine-label">ENGINE</label>
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
          <span className="header-cycle mono">CYC <span className="text-cyan">{state.cycleNumber}</span></span>
        )}
      </div>
    </header>
  )
}
