import { type ChangeEvent, type FormEvent, type MouseEvent as ReactMouseEvent, useEffect, useRef, useState } from 'react'

type Zone = {
  zone_id: string
  name: string
  density_persons_m2: number
  headcount: number
  capacity_percent: number
  level: 'green' | 'yellow' | 'amber' | 'red'
  confidence: number
}

type Alert = {
  id: string
  zone_id: string
  level: Zone['level']
  reading: string
  recommendation: string
  acknowledged: boolean
  escalated?: boolean
  created_at: string
}
type DemoState = {
  scenario: string
  scenario_label: string
  running: boolean
  tick: number
  zones: Zone[]
  alerts: Alert[]
  synthetic: true
}
type User = { username: string; role: string }
type AuditEvent = {
  event_id: string
  actor: string
  action: string
  object_id: string
  occurred_at: string
  details: string
}
type Retention = { logs_days: number; clips_days: number; unmatched_face_hours: number }
type WatchlistAuthorization = {
  authorization_id: string; reference: string; authorized_by: string
  created_at: string; expires_at: string; revoked_at?: string | null; active: boolean
}
type Point = { x: number; y: number }
type DrawnZone = { zone_id: string; name: string; points: Point[] }
type GroundReference = { corners: Point[]; width_m: number; depth_m: number }
type CalibrationResult = {
  status: 'uncalibrated' | 'invalid' | 'experimental-area-mapping'
  reference_area_m2: number | null
  zones: { zone_id: string; name: string; area_m2: number | null; status: string }[]
  notice?: string
}
type CameraSetup = {
  camera_name: string; source_name: string; frame_width: number; frame_height: number
  ground_reference: GroundReference | null; zones: DrawnZone[]
  calibration?: CalibrationResult
  updated_by: string; updated_at: string; calibration_status: 'draft' | 'uncalibrated'
}
type DrawingMode = 'zone' | 'ground' | null

const demoZones: Zone[] = [
  { zone_id: 'north-gate', name: 'North Gate', density_persons_m2: 1.1, headcount: 462, capacity_percent: 16.9, level: 'green', confidence: 0 },
  { zone_id: 'main-plaza', name: 'Main Plaza', density_persons_m2: 0.7, headcount: 392, capacity_percent: 12.7, level: 'green', confidence: 0 },
]

function formatTime(value: number) {
  if (!Number.isFinite(value)) return '00:00'
  const minutes = Math.floor(value / 60).toString().padStart(2, '0')
  const seconds = Math.floor(value % 60).toString().padStart(2, '0')
  return `${minutes}:${seconds}`
}

export default function App() {
  const [zones, setZones] = useState<Zone[]>(demoZones)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [connected, setConnected] = useState(false)
  const [user, setUser] = useState<User | null>(null)
  const [loginError, setLoginError] = useState('')
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([])
  const [auditError, setAuditError] = useState('')
  const [auditSearch, setAuditSearch] = useState('')
  const [auditAction, setAuditAction] = useState('')
  const [auditStart, setAuditStart] = useState('')
  const [auditEnd, setAuditEnd] = useState('')
  const [appliedAuditFilters, setAppliedAuditFilters] = useState('')
  const [retention, setRetention] = useState<Retention>({ logs_days: 7, clips_days: 7, unmatched_face_hours: 24 })
  const [retentionMessage, setRetentionMessage] = useState('')
  const [authorizations, setAuthorizations] = useState<WatchlistAuthorization[]>([])
  const [authorizationMessage, setAuthorizationMessage] = useState('')
  const [replayUrl, setReplayUrl] = useState('')
  const [replayName, setReplayName] = useState('')
  const [demoState, setDemoState] = useState<DemoState | null>(null)
  const [demoMessage, setDemoMessage] = useState('')
  const [replayPlaying, setReplayPlaying] = useState(false)
  const [replayTime, setReplayTime] = useState(0)
  const [replayDuration, setReplayDuration] = useState(0)
  const [replaySpeed, setReplaySpeed] = useState(1)
  const [replayLoop, setReplayLoop] = useState(true)
  const [frameSize, setFrameSize] = useState({ width: 0, height: 0 })
  const [cameraName, setCameraName] = useState('Local test camera')
  const [cameraSetup, setCameraSetup] = useState<CameraSetup | null>(null)
  const [drawMode, setDrawMode] = useState<DrawingMode>(null)
  const [drawPoints, setDrawPoints] = useState<Point[]>([])
  const [drawnZones, setDrawnZones] = useState<DrawnZone[]>([])
  const [zoneName, setZoneName] = useState('')
  const [groundWidth, setGroundWidth] = useState('')
  const [groundDepth, setGroundDepth] = useState('')
  const [groundReference, setGroundReference] = useState<GroundReference | null>(null)
  const [setupMessage, setSetupMessage] = useState('')
  const videoRef = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    fetch('/api/auth/me').then(async (r) => {
      if (!r.ok) return
      const current = await r.json() as User
      setUser(current)
      const simulationResponse = await fetch('/api/demo/state')
      if (!simulationResponse.ok) throw new Error()
      const currentSimulation = await simulationResponse.json() as DemoState
      setDemoState(currentSimulation)
      setZones(currentSimulation.zones)
      setAlerts(currentSimulation.alerts)
      const setupResponse = await fetch('/api/camera/setup')
      if (setupResponse.ok) {
        const savedSetup = await setupResponse.json() as CameraSetup | null
        if (savedSetup) {
          setCameraSetup(savedSetup)
          setCameraName(savedSetup.camera_name)
          setDrawnZones(savedSetup.zones)
          setGroundReference(savedSetup.ground_reference)
          if (savedSetup.ground_reference) {
            setGroundWidth(String(savedSetup.ground_reference.width_m))
            setGroundDepth(String(savedSetup.ground_reference.depth_m))
          }
        }
      }
      if (current.role === 'supervisor' || current.role === 'administrator') {
        const auditResponse = await fetch('/api/audit')
        if (auditResponse.ok) setAuditEvents(await auditResponse.json() as AuditEvent[])
        else setAuditError('Audit log could not be loaded.')
      }
      if (current.role === 'administrator') {
        const [retentionResponse, authorizationResponse] = await Promise.all([
          fetch('/api/admin/retention'), fetch('/api/admin/watchlist-authorizations'),
        ])
        if (retentionResponse.ok) setRetention(await retentionResponse.json() as Retention)
        if (authorizationResponse.ok) setAuthorizations(await authorizationResponse.json() as WatchlistAuthorization[])
      }
      setConnected(true)
    }).catch(() => setConnected(false))
  }, [])

  useEffect(() => {
    if (!user) return
    let active = true
    const refreshReadings = async () => {
      try {
        const response = await fetch('/api/demo/tick', { method: 'POST' })
        if (response.status === 401) {
          if (active) setUser(null)
          return
        }
        if (!response.ok) throw new Error()
        const nextSimulation = await response.json() as DemoState
        if (!active) return
        setDemoState(nextSimulation)
        setZones(nextSimulation.zones)
        setAlerts(nextSimulation.alerts)
        setConnected(true)
      } catch {
        if (active) setConnected(false)
      }
    }
    const timer = window.setInterval(() => { void refreshReadings() }, 2000)
    return () => { active = false; window.clearInterval(timer) }
  }, [user])

  useEffect(() => () => {
    if (replayUrl) URL.revokeObjectURL(replayUrl)
  }, [replayUrl])

  function loadReplay(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0]
    if (!file) return
    const hasDifferentSavedSource = Boolean(cameraSetup?.source_name && cameraSetup.source_name !== file.name)
    setReplayUrl(URL.createObjectURL(file))
    setReplayName(file.name)
    setReplayPlaying(false)
    setReplayTime(0)
    setReplayDuration(0)
    setFrameSize({ width: 0, height: 0 })
    if (hasDifferentSavedSource) {
      setCameraSetup(null)
      setDrawnZones([])
      setGroundReference(null)
      setGroundWidth('')
      setGroundDepth('')
      setSetupMessage('This recording differs from the saved setup. Start a new drawing for this video.')
    } else {
      setSetupMessage('Recording loaded locally. It is not uploaded or analyzed.')
    }
    setDrawMode(null)
    setDrawPoints([])
    event.currentTarget.value = ''
  }

  function beginDrawing(mode: Exclude<DrawingMode, null>) {
    if (!replayUrl) return
    videoRef.current?.pause()
    setDrawPoints([])
    setDrawMode(mode)
    setSetupMessage(mode === 'zone'
      ? 'Click at least three points around one zone, then add its name.'
      : 'Click four corners of a known flat ground rectangle, in order around its edge.')
  }

  function addDrawingPoint(event: ReactMouseEvent<SVGSVGElement>) {
    if (!drawMode) return
    const rect = event.currentTarget.getBoundingClientRect()
    const point = {
      x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
    }
    setDrawPoints((existing) => [...existing, point])
    const pointCount = drawPoints.length + 1
    setSetupMessage(drawMode === 'ground'
      ? `Ground corners selected: ${pointCount}/4. Click the next corner around the same flat rectangle.`
      : `Zone corners selected: ${pointCount}. Click at least three points around the zone.`)
  }

  function finishGroundReference() {
    if (drawPoints.length !== 4) {
      setSetupMessage('Mark exactly four ground corners first.')
      return
    }
    const width = Number(groundWidth)
    const depth = Number(groundDepth)
    if (!(width > 0 && depth > 0)) {
      setSetupMessage('Enter the real width and depth of the marked ground rectangle.')
      return
    }
    setGroundReference({ corners: drawPoints, width_m: width, depth_m: depth })
    setCameraSetup(null)
    setDrawPoints([])
    setDrawMode(null)
    setSetupMessage('Ground reference saved in this form. Now draw one or more zones, then save the camera setup.')
  }

  function finishZone() {
    if (drawPoints.length < 3 || !zoneName.trim()) {
      setSetupMessage('A zone needs at least three points and a name.')
      return
    }
    setDrawnZones((existing) => [...existing, {
      zone_id: crypto.randomUUID(), name: zoneName.trim(), points: drawPoints,
    }])
    setCameraSetup(null)
    setZoneName('')
    setDrawPoints([])
    setDrawMode(null)
    setSetupMessage('Zone added to this unsaved setup. Save setup to keep it.')
  }

  async function saveCameraSetup() {
    if (!replayUrl || !frameSize.width || !frameSize.height) {
      setSetupMessage('Load a video and wait for it to finish opening before saving.')
      return
    }
    try {
      const response = await fetch('/api/camera/setup', {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          camera_name: cameraName.trim() || 'Local test camera',
          source_name: replayName,
          frame_width: frameSize.width,
          frame_height: frameSize.height,
          ground_reference: groundReference,
          zones: drawnZones,
        }),
      })
      if (!response.ok) {
        const detail = await response.json().catch(() => null) as { detail?: string } | null
        throw new Error(detail?.detail ?? 'Could not save setup.')
      }
      const saved = await response.json() as CameraSetup
      setCameraSetup(saved)
      setCameraName(saved.camera_name)
      setSetupMessage(saved.ground_reference
        ? `Setup saved. ${saved.calibration?.zones?.filter((zone) => zone.area_m2 != null).length ?? 0} zone area estimate(s) calculated. Dashboard readings are still simulated.`
        : 'Zones saved. Add a ground reference to estimate zone area. Dashboard readings are still simulated.')
    } catch (error) {
      setSetupMessage(error instanceof Error ? error.message : 'Could not save setup. Check the backend connection.')
    }
  }

  async function startDemoScenario(scenario: string) {
    setDemoMessage('')
    try {
      const response = await fetch('/api/demo/scenario', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario }),
      })
      if (!response.ok) throw new Error()
      const next = await response.json() as DemoState
      setDemoState(next)
      setZones(next.zones)
      setAlerts(next.alerts)
      setDemoMessage(`${next.scenario_label} simulation started. These are made-up readings, not AI results.`)
    } catch {
      setDemoMessage('Could not start the scenario. Check the backend connection.')
    }
  }

  async function stopDemoScenario() {
    setDemoMessage('')
    try {
      const response = await fetch('/api/demo/stop', { method: 'POST' })
      if (!response.ok) throw new Error()
      const next = await response.json() as DemoState
      setDemoState(next)
      setZones(next.zones)
      setAlerts(next.alerts)
      setDemoMessage('Simulation stopped. Readings returned to the steady demo baseline.')
    } catch {
      setDemoMessage('Could not stop the scenario. Check the backend connection.')
    }
  }

  async function toggleReplay() {
    const video = videoRef.current
    if (!video) return
    if (video.paused) await video.play()
    else video.pause()
  }

  async function signIn(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const form = new FormData(event.currentTarget)
    setLoginError('')
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: form.get('username'), password: form.get('password') }),
      })
      if (response.status === 401) { setLoginError('Sign-in failed. Check the username and password.'); return }
      if (!response.ok) { setLoginError(`Sign-in service error (${response.status}). Check the backend terminal.`); return }
      window.location.reload()
    } catch {
      setLoginError('Cannot reach the backend. Start it in a terminal, then try again.')
    }
  }

  async function signOut() {
    await fetch('/api/auth/logout', { method: 'POST' })
    setUser(null)
    setConnected(false)
  }

  async function refreshAudit(filters = appliedAuditFilters) {
    setAuditError('')
    try {
      const suffix = filters ? `?${filters}` : ''
      const response = await fetch(`/api/audit${suffix}`)
      if (!response.ok) throw new Error()
      setAuditEvents(await response.json() as AuditEvent[])
    } catch {
      setAuditError('Audit log could not be loaded. Check your role and backend connection.')
    }
  }

  function auditExportUrl(format: 'csv' | 'pdf') {
    return `/api/audit/export.${format}${appliedAuditFilters ? `?${appliedAuditFilters}` : ''}`
  }

  async function saveRetention(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const response = await fetch('/api/admin/retention', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(retention),
    })
    setRetentionMessage(response.ok ? 'Retention settings saved.' : 'Could not save settings. Check the allowed ranges.')
    if (response.ok) setRetention(await response.json() as Retention)
  }

  async function createAuthorization(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const formElement = event.currentTarget
    const form = new FormData(formElement)
    const response = await fetch('/api/admin/watchlist-authorizations', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reference: form.get('authorizationReference'), expires_at: new Date(String(form.get('authorizationExpiry'))).toISOString() }),
    })
    if (!response.ok) { setAuthorizationMessage('Could not record authorization. Check the reference and expiry.'); return }
    setAuthorizationMessage('Written authorization recorded. Watchlist matching is not connected in this software increment.')
    formElement.reset()
    const rows = await fetch('/api/admin/watchlist-authorizations')
    if (rows.ok) setAuthorizations(await rows.json() as WatchlistAuthorization[])
    await refreshAudit()
  }

  async function revokeAuthorization(id: string) {
    const response = await fetch(`/api/admin/watchlist-authorizations/${id}`, { method: 'DELETE' })
    setAuthorizationMessage(response.ok ? 'Authorization revoked.' : 'Could not revoke authorization.')
    const rows = await fetch('/api/admin/watchlist-authorizations')
    if (rows.ok) setAuthorizations(await rows.json() as WatchlistAuthorization[])
    await refreshAudit()
  }

  async function acknowledge(id: string) {
    try {
      const response = await fetch(`/api/alerts/${id}/acknowledge`, { method: 'POST' })
      if (!response.ok) throw new Error()
      const updated = await response.json() as Alert
      setAlerts((items) => items.map((item) => item.id === id ? updated : item))
    } catch {
      setLoginError('Could not save acknowledgement. Check the backend connection and sign-in.')
    }
  }

  if (!user) return <main className="login-shell"><form className="login-card" onSubmit={signIn}>
    <span className="eyebrow">SAKAR ROBOTICS · LTV</span><h1>Sign in</h1>
    <p>Use your assigned role account to open the operations dashboard.</p>
    <label>Username<input name="username" autoComplete="username" required /></label>
    <label>Password<input name="password" type="password" autoComplete="current-password" required /></label>
    {loginError && <div className="login-error">{loginError}</div>}
    <button type="submit">Continue</button>
    <small>Local development accounts are created by the backend on first start.</small>
  </form></main>

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#overview"><span className="brand-mark">L</span><span><b>LOOKOUT</b><small>VEHICLE OPERATIONS</small></span></a>
      <div className="nav-label">WORKSPACE</div>
      <nav className="side-nav" aria-label="Main navigation">
        <a className="nav-item active" href="#overview"><span className="nav-icon">⌂</span>Overview</a>
        <a className="nav-item" href="#zones"><span className="nav-icon">◫</span>Zones</a>
        <a className="nav-item" href="#camera-setup"><span className="nav-icon">⌖</span>Camera setup</a>
        <a className="nav-item" href="#alerts"><span className="nav-icon">◉</span>Alerts<span className="nav-count">{alerts.filter((a) => !a.acknowledged).length}</span></a>
        {(user.role === 'supervisor' || user.role === 'administrator') && <a className="nav-item" href="#audit"><span className="nav-icon">▤</span>Audit log</a>}
        {user.role === 'administrator' && <a className="nav-item" href="#admin"><span className="nav-icon">⚙</span>Administration</a>}
      </nav>
      <div className="sidebar-bottom"><span className="user-avatar">{user.username.slice(0, 1).toUpperCase()}</span><span className="user-summary"><b>{user.username}</b><small>{user.role}</small></span><button className="sidebar-signout" onClick={signOut} aria-label="Sign out" title="Sign out">↪</button></div>
    </aside>
    <main className="workspace">
    <header id="overview" className="topbar">
      <div><div className="eyebrow">SAKAR ROBOTICS / OPERATIONS</div><h1>Operations overview</h1></div>
      <div className="header-actions"><div className="site-badge"><span className="site-indicator"/>LOCAL SIMULATION <small>Not connected to a vehicle</small></div><div className="status"><span className={`dot ${connected ? 'live' : ''}`} />{connected ? 'Backend connected' : 'Backend disconnected'}</div></div>
    </header>
    <section className="summary">
      <article><div className="metric-top"><span className="label">ACTIVE ZONES</span><span className="metric-icon blue">◫</span></div><strong>{zones.length}</strong><small>All sectors monitored</small></article>
      <article><div className="metric-top"><span className="label">PEAK DENSITY</span><span className="metric-icon violet">◉</span></div><strong>{Math.max(...zones.map((z) => z.density_persons_m2)).toFixed(1)} <em>people/m²</em></strong><small>Across configured zones</small></article>
      <article><div className="metric-top"><span className="label">TOTAL HEADCOUNT</span><span className="metric-icon green">Σ</span></div><strong>{zones.reduce((sum, zone) => sum + zone.headcount, 0).toLocaleString()}</strong><small>Estimated across zones</small></article>
      <article><div className="metric-top"><span className="label">OPEN ALERTS</span><span className="metric-icon amber">!</span></div><strong>{alerts.filter((a) => !a.acknowledged).length}</strong><small>Officer review required</small></article>
    </section>
    <div className="grid">
      <section className="panel panorama">
        <div className="panel-head"><div><span className="eyebrow">DEVELOPMENT MODE · LOCAL REPLAY</span><h2>Recorded camera view</h2></div><label className="video-open">Load test recording<input type="file" accept="video/*" onChange={loadReplay}/></label></div>
        {replayUrl ? <><div className="replay-frame" style={{ aspectRatio: frameSize.width ? `${frameSize.width} / ${frameSize.height}` : '16 / 9', width: frameSize.width ? `min(100%, ${Math.round(500 * frameSize.width / frameSize.height)}px)` : '100%' }}><video ref={videoRef} src={replayUrl} controls preload="metadata" loop={replayLoop} onPlay={() => setReplayPlaying(true)} onPause={() => setReplayPlaying(false)} onLoadedMetadata={(event) => { setReplayDuration(event.currentTarget.duration); setFrameSize({ width: event.currentTarget.videoWidth, height: event.currentTarget.videoHeight }) }} onTimeUpdate={(event) => setReplayTime(event.currentTarget.currentTime)}>Your browser cannot play this video format.</video><svg className={`drawing-overlay ${drawMode ? 'active' : ''}`} viewBox="0 0 1000 1000" preserveAspectRatio="none" onClick={addDrawingPoint} aria-label="Camera setup drawing area">
            {drawnZones.map((zone) => <g key={zone.zone_id}><polygon points={zone.points.map((point) => `${point.x * 1000},${point.y * 1000}`).join(' ')} /><text x={zone.points[0].x * 1000} y={zone.points[0].y * 1000}>{zone.name}</text></g>)}
            {groundReference && <polygon className="ground-polygon" points={groundReference.corners.map((point) => `${point.x * 1000},${point.y * 1000}`).join(' ')} />}
            {drawPoints.length > 0 && <><polyline className="draft-line" points={drawPoints.map((point) => `${point.x * 1000},${point.y * 1000}`).join(' ')} />{drawPoints.map((point, index) => <circle key={index} cx={point.x * 1000} cy={point.y * 1000} r="10" />)}</>}
          </svg><div className="replay-file">{replayName} · local only</div></div>
          <div className="replay-controls"><button className="filter-button" onClick={() => void toggleReplay()}>{replayPlaying ? 'Pause replay' : 'Play replay'}</button><span>{formatTime(replayTime)} / {formatTime(replayDuration)}</span><label>Speed<select value={replaySpeed} onChange={(event) => { const speed = Number(event.target.value); setReplaySpeed(speed); if (videoRef.current) videoRef.current.playbackRate = speed }}><option value={0.5}>0.5×</option><option value={1}>1×</option><option value={1.5}>1.5×</option><option value={2}>2×</option></select></label><label className="loop-control"><input type="checkbox" checked={replayLoop} onChange={(event) => setReplayLoop(event.target.checked)} /> Repeat</label></div></>
          : <div className="scene"><div className="horizon"/><div className="sun"/><div className="scene-copy"><span>NO RECORDING SELECTED</span><b>Load a local video to use it as a replay</b></div><div className="sector s1">NORTH</div><div className="sector s2">EAST</div><div className="sector s3">SOUTH</div><div className="sector s4">WEST</div></div>}
        <div className="scene-foot"><span><i className="legend blue"/> Replay plays locally in this browser; it is not uploaded.</span><span>Video playback is not analyzed by AI.</span></div>
        <section id="camera-setup" className="camera-setup">
          <div className="setup-heading"><div><span className="eyebrow">CONFIGURATION · DRAFT ONLY</span><h3>Camera and zone setup</h3></div><span className={`setup-status ${cameraSetup?.calibration_status ?? 'uncalibrated'}`}>{cameraSetup?.calibration_status === 'draft' ? 'Calibration draft' : 'Not calibrated'}</span></div>
          <p className="setup-note">Use a paused recording to mark zones and a flat ground rectangle whose real width and depth you know. The software uses that reference to estimate zone areas. The video stays in your browser. This does not count people or change the synthetic dashboard readings.</p>
          <div className="setup-fields"><label>Camera label<input value={cameraName} maxLength={100} onChange={(event) => setCameraName(event.target.value)} placeholder="For example: Main plaza camera" /></label><span>Recording: {replayName || cameraSetup?.source_name || 'None loaded'}</span></div>
          <div className="setup-actions"><button className="filter-button" disabled={!replayUrl} onClick={() => beginDrawing('zone')}>Draw a zone</button><button className="setup-secondary" disabled={!replayUrl} onClick={() => beginDrawing('ground')}>Mark ground reference</button>{drawMode && <button className="setup-secondary" onClick={() => { setDrawMode(null); setDrawPoints([]); setSetupMessage('Drawing cancelled.') }}>Cancel drawing</button>}</div>
          {drawMode === 'zone' && <div className="drawing-fields"><p className="drawing-hint">Click the video to mark at least 3 corners around the zone. Selected points appear as circles on the video.</p><div className="setup-fields"><label>Zone name<input value={zoneName} maxLength={80} onChange={(event) => setZoneName(event.target.value)} placeholder="For example: North Gate" /></label><span className="point-count">Corners marked: {drawPoints.length}</span><button className="setup-secondary" disabled={!drawPoints.length} onClick={() => { setDrawPoints((points) => points.slice(0, -1)); setSetupMessage('Removed the last point.') }}>Undo last point</button><button className="setup-secondary" disabled={drawPoints.length < 3} onClick={finishZone}>Add zone</button></div></div>}
          {drawMode === 'ground' && <div className="drawing-fields"><p className="drawing-hint">First click four corners on the video, going around one flat rectangle in order. The button stays disabled until it shows 4/4.</p><div className="setup-fields"><label>Real ground width (metres)<input type="number" min="0.1" step="0.1" value={groundWidth} onChange={(event) => setGroundWidth(event.target.value)} /></label><label>Real ground depth (metres)<input type="number" min="0.1" step="0.1" value={groundDepth} onChange={(event) => setGroundDepth(event.target.value)} /></label><span className="point-count">Corners marked: {drawPoints.length}/4</span><button className="setup-secondary" disabled={!drawPoints.length} onClick={() => { setDrawPoints((points) => points.slice(0, -1)); setSetupMessage('Removed the last point.') }}>Undo last point</button><button className="setup-secondary" disabled={drawPoints.length !== 4} onClick={finishGroundReference}>Save ground reference</button></div></div>}
          {drawnZones.length > 0 && <div className="drawn-zone-list">{drawnZones.map((zone) => {
            const area = cameraSetup?.calibration?.zones?.find((item) => item.zone_id === zone.zone_id)
            return <div key={zone.zone_id}><span>{zone.name}<small>{zone.points.length} corners{area?.area_m2 != null ? ` · about ${area.area_m2} m²` : ''}</small></span><button className="text-button" onClick={() => { setDrawnZones((existing) => existing.filter((item) => item.zone_id !== zone.zone_id)); setCameraSetup(null) }}>Remove</button></div>
          })}</div>}
          {cameraSetup?.calibration?.notice && <p className="setup-note" role="status">{cameraSetup.calibration.notice} These areas are for setup review only, not operational readings.</p>}
          {cameraSetup && !cameraSetup.calibration && <p className="setup-note" role="status">This setup was saved by an older backend version. Restart the backend, then refresh this page to calculate area estimates.</p>}
          {groundReference && <p className="setup-note">Marked ground reference: {groundReference.width_m} m × {groundReference.depth_m} m · draft, not validated</p>}
          {cameraSetup?.calibration && <div className="calibration-result" role="status"><b>Saved calibration estimate</b><span>Reference area: {cameraSetup.calibration.reference_area_m2 ?? 'not available'} m²</span>{cameraSetup.calibration.zones.length === 0 ? <span>Add a zone inside the marked rectangle to see its area here.</span> : cameraSetup.calibration.zones.map((zone) => <span key={zone.zone_id}>{zone.name}: {zone.area_m2 != null ? `about ${zone.area_m2} m²` : 'area unavailable; check that it is inside the reference'}</span>)}</div>}
          <div className="setup-footer"><button className="filter-button" disabled={!replayUrl} onClick={() => void saveCameraSetup()}>Save camera setup</button>{cameraSetup && <span>Last saved by {cameraSetup.updated_by} · {new Date(cameraSetup.updated_at).toLocaleString()}</span>}</div>
          {setupMessage && <p className="audit-message" role="status">{setupMessage}</p>}
        </section>
        <div className="simulation-controls"><div><span className="eyebrow">SYNTHETIC CROWD READINGS · NOT AI</span><h3>Run a dashboard scenario</h3><p>These controls change sample zone readings and create sample alerts so you can see how the dashboard responds.</p></div><div className="scenario-buttons"><button onClick={() => void startDemoScenario('steady')}>Steady</button><button onClick={() => void startDemoScenario('rising')}>Crowd rising</button><button onClick={() => void startDemoScenario('surge')}>Surge</button><button onClick={() => void startDemoScenario('critical')}>Critical density</button><button className="stop-scenario" onClick={() => void stopDemoScenario()}>Stop</button></div><div className="simulation-status"><span>{demoState?.running ? `Running: ${demoState.scenario_label} · update ${demoState.tick}` : 'Stopped · steady baseline'}</span>{demoMessage && <span role="status">{demoMessage}</span>}</div></div>
      </section>
      <section id="alerts" className="panel alerts">
        <div className="panel-head"><div><span className="eyebrow">NEEDS ATTENTION</span><h2>Alert queue</h2></div><span className="count">{alerts.length}</span></div>
        <div className="alert-list">{alerts.length === 0 ? <p className="audit-message">No active alerts in the steady demo scenario.</p> : alerts.map((alert) => <article className="alert-card" key={alert.id}>
          <div className="alert-top"><span className={`tag ${alert.level}`}>{alert.level}</span><time>{new Date(alert.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time></div>
          <h3>{zones.find((z) => z.zone_id === alert.zone_id)?.name ?? alert.zone_id}</h3>
          <p className="reading">{alert.reading}</p><p>{alert.recommendation}</p>{alert.escalated && <p className="escalated-note">Escalated for supervisor review</p>}
          <button disabled={alert.acknowledged} onClick={() => acknowledge(alert.id)}>{alert.acknowledged ? 'Acknowledged' : 'Acknowledge alert'}</button>
        </article>)}</div>
      <p className="human-note">Demo readings only. AI outputs are recommendations; officers decide every action.</p>
      </section>
    </div>
    <section id="zones" className="panel zone-panel"><div className="panel-head"><div><span className="eyebrow">SITE MONITORING</span><h2>Zone readings</h2></div><span className="subtle">Refreshes every 2 seconds</span></div>
      <div className="zone-table"><div className="table-row table-header"><span>Zone</span><span>Density</span><span>Headcount</span><span>Capacity</span><span>Confidence</span><span>State</span></div>
        {zones.map((zone) => <div className="table-row" key={zone.zone_id}><strong>{zone.name}</strong><span>{zone.density_persons_m2.toFixed(1)} people/m²</span><span>{zone.headcount.toLocaleString()}</span><span><span className="bar"><i style={{ width: `${zone.capacity_percent}%` }}/></span>{zone.capacity_percent}%</span><span>{Math.round(zone.confidence * 100)}%</span><span className={`state ${zone.level}`}>{zone.level}</span></div>)}
      </div>
    </section>
    {(user.role === 'supervisor' || user.role === 'administrator') && <section id="audit" className="panel zone-panel audit-panel">
      <div className="panel-head"><div><span className="eyebrow">ACCESS CONTROLLED</span><h2>Audit log</h2></div><div className="export-actions"><a href={auditExportUrl('csv')}>Download CSV</a><a href={auditExportUrl('pdf')}>Download PDF</a></div></div>
      <p className="audit-note">Search events and export matching results (up to 5,000 events). The log is read-only in this dashboard.</p>
      <form className="audit-filters" onSubmit={(event) => {
        event.preventDefault()
        const params = new URLSearchParams()
        if (auditSearch.trim()) params.set('q', auditSearch.trim())
        if (auditAction) params.set('action', auditAction)
        if (auditStart) params.set('start', new Date(auditStart).toISOString())
        if (auditEnd) params.set('end', new Date(auditEnd).toISOString())
        const query = params.toString()
        setAppliedAuditFilters(query)
        void refreshAudit(query)
      }}>
        <input aria-label="Search audit log" placeholder="Search user, action or record" value={auditSearch} onChange={(event) => setAuditSearch(event.target.value)} />
        <select aria-label="Filter by action" value={auditAction} onChange={(event) => setAuditAction(event.target.value)}><option value="">All actions</option><option value="auth.login">Sign in</option><option value="auth.logout">Sign out</option><option value="demo.scenario.start">Demo scenario started</option><option value="demo.scenario.stop">Demo scenario stopped</option><option value="alert.created">Alert created</option><option value="alert.acknowledge">Alert acknowledged</option><option value="alert.escalated">Alert escalated</option><option value="retention.update">Retention changed</option><option value="watchlist.authorization.create">Watchlist authorization</option><option value="watchlist.authorization.revoke">Authorization revoked</option></select>
        <label>From<input type="datetime-local" value={auditStart} onChange={(event) => setAuditStart(event.target.value)} /></label>
        <label>To<input type="datetime-local" value={auditEnd} onChange={(event) => setAuditEnd(event.target.value)} /></label>
        <button className="filter-button" type="submit">Apply filters</button>
      </form>
      {auditError ? <p className="audit-message">{auditError}</p> : auditEvents.length === 0 ? <p className="audit-message">No events recorded yet.</p> : <div className="audit-table">
        <div className="audit-row audit-header"><span>Time</span><span>User</span><span>Action</span><span>Record</span><span>Details</span></div>
        {auditEvents.map((event) => <div className="audit-row" key={event.event_id}>
          <time>{new Date(event.occurred_at).toLocaleString()}</time><span>{event.actor}</span><span>{event.action}</span><span>{event.object_id}</span><span>{event.details}</span>
        </div>)}
      </div>}
    </section>}
    {user.role === 'administrator' && <section id="admin" className="panel zone-panel admin-settings">
      <div className="panel-head"><div><span className="eyebrow">ADMINISTRATOR ONLY</span><h2>Privacy and retention settings</h2></div></div>
      <form className="settings-form" onSubmit={saveRetention}>
        <label>Log retention (days)<input type="number" min="1" max="30" value={retention.logs_days} onChange={(event) => setRetention({ ...retention, logs_days: Number(event.target.value) })} /></label>
        <label>Clip retention (days)<input type="number" min="1" max="30" value={retention.clips_days} onChange={(event) => setRetention({ ...retention, clips_days: Number(event.target.value) })} /></label>
        <label>Unmatched face image retention (hours)<input type="number" min="1" max="168" value={retention.unmatched_face_hours} onChange={(event) => setRetention({ ...retention, unmatched_face_hours: Number(event.target.value) })} /></label>
        <button className="filter-button" type="submit">Save retention settings</button>
      </form>
      <p className="audit-note">Log retention is enforced. Clip and face-image retention values are saved for the later video and face-data components.</p>
      {retentionMessage && <p className="audit-message">{retentionMessage}</p>}
      <div className="watchlist-admin"><h3>Watchlist authorization</h3><p className="audit-note">Off by default. Record the written authorization and expiry. This scaffold does not perform watchlist matching.</p>
        <form className="settings-form" onSubmit={createAuthorization}>
          <label>Written authorization reference<input name="authorizationReference" required maxLength={200} placeholder="Department order or record reference" /></label>
          <label>Expires at<input name="authorizationExpiry" type="datetime-local" required /></label>
          <button className="filter-button" type="submit">Record authorization</button>
        </form>
        {authorizationMessage && <p className="audit-message">{authorizationMessage}</p>}
        {authorizations.length > 0 && <div className="authorization-list">{authorizations.map((authorization) => <div className="authorization-row" key={authorization.authorization_id}>
          <span><b>{authorization.reference}</b><small>By {authorization.authorized_by} · expires {new Date(authorization.expires_at).toLocaleString()}</small></span>
          <span className={authorization.active ? 'state green' : 'state red'}>{authorization.active ? 'Active' : authorization.revoked_at ? 'Revoked' : 'Expired'}</span>
          {authorization.active && <button className="signout" onClick={() => revokeAuthorization(authorization.authorization_id)}>Revoke</button>}
        </div>)}</div>}
      </div>
    </section>}
    <footer><span>LTV SOFTWARE FOUNDATION · 0.1</span><span>NO ANNOUNCEMENTS ARE SENT BY THIS SOFTWARE</span></footer>
    </main>
  </div>
}
