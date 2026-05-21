import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { SimulationProvider } from './context/SimulationContext'
import Header              from './components/Layout/Header'
import Dashboard           from './components/Dashboard'
import PreviousSimulations from './components/History/PreviousSimulations'
import { useHashRoute }    from './hooks/useHashRoute'
import './styles/App.css'

function AppContent() {
  const rootRef = useRef(null)
  const { path } = useHashRoute()

  // Boot sequence: scan line reveal
  useEffect(() => {
    const el = rootRef.current
    if (!el) return
    gsap.from(el, { duration: 0.3, ease: 'power2.out', clearProps: 'all' })
  }, [])

  const showHistory = path.startsWith('/history')

  // Dashboard is kept mounted so live simulation state is preserved when
  // the user toggles to Previous Simulations and back.
  return (
    <div ref={rootRef} className="app-root">
      <Header />
      <div className="app-body">
        <div style={{ display: showHistory ? 'none' : 'contents' }}>
          <Dashboard />
        </div>
        {showHistory && <PreviousSimulations />}
      </div>
    </div>
  )
}

export default function App() {
  return (
    <SimulationProvider>
      <AppContent />
    </SimulationProvider>
  )
}
