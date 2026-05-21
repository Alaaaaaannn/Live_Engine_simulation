import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import { SimulationProvider }      from './context/SimulationContext'
import { AuthProvider, useAuth }   from './context/AuthContext'
import Header              from './components/Layout/Header'
import Dashboard           from './components/Dashboard'
import PreviousSimulations from './components/History/PreviousSimulations'
import LoginPage          from './components/Auth/LoginPage'
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

function Gate() {
  const { user, bootstrapping } = useAuth()

  // Listen for 401 events from the axios interceptor and force a remount
  // by reloading — simplest way to flush in-flight simulation state.
  useEffect(() => {
    const onExpired = () => window.location.reload()
    window.addEventListener('dt:auth-expired', onExpired)
    return () => window.removeEventListener('dt:auth-expired', onExpired)
  }, [])

  if (bootstrapping) return <div className="auth-shell" />
  if (!user) return <LoginPage />

  return (
    <SimulationProvider>
      <AppContent />
    </SimulationProvider>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <Gate />
    </AuthProvider>
  )
}
