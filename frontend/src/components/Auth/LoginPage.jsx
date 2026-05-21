import { useState } from 'react'
import { useAuth } from '../../context/AuthContext'
import './Auth.css'

export default function LoginPage() {
  const { login, register } = useAuth()
  const [mode, setMode]         = useState('login')   // 'login' | 'register'
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy]         = useState(false)
  const [error, setError]       = useState('')

  const submit = async (e) => {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      if (mode === 'login') await login(email.trim().toLowerCase(), password)
      else                  await register(email.trim().toLowerCase(), password)
    } catch (err) {
      const detail = err?.response?.data?.detail
      setError(typeof detail === 'string' ? detail : 'Authentication failed.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-shell">
      <div className="auth-card">
        <h2 className="auth-title">Digital Twin</h2>
        <p className="auth-sub">{mode === 'login' ? 'Sign in to continue' : 'Create your account'}</p>

        <form onSubmit={submit} className="auth-form">
          <label className="auth-label">
            Email
            <input
              type="email" required autoComplete="email"
              value={email} onChange={e => setEmail(e.target.value)}
              placeholder="you@example.com"
            />
          </label>

          <label className="auth-label">
            Password
            <input
              type="password" required minLength={6}
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              value={password} onChange={e => setPassword(e.target.value)}
              placeholder="At least 6 characters"
            />
          </label>

          {error && <div className="auth-error">{error}</div>}

          <button type="submit" className="auth-submit" disabled={busy}>
            {busy ? '…' : (mode === 'login' ? 'Sign in' : 'Register')}
          </button>
        </form>

        <button
          type="button"
          className="auth-switch"
          onClick={() => { setError(''); setMode(mode === 'login' ? 'register' : 'login') }}
        >
          {mode === 'login'
            ? "Don't have an account? Register"
            : 'Already registered? Sign in'}
        </button>
      </div>
    </div>
  )
}
