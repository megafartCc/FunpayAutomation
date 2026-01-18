import { useEffect, useMemo, useState } from 'react'
import './index.css'

const api = async (path, options = {}) => {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  if (!res.ok) {
    let detail
    try {
      const data = await res.json()
      detail = data.detail || JSON.stringify(data)
    } catch {
      detail = await res.text()
    }
    throw new Error(detail || res.statusText)
  }
  const text = await res.text()
  try {
    return JSON.parse(text || '{}')
  } catch {
    return text
  }
}

export default function App() {
  const [session, setSession] = useState({ polling: false, userId: null, baseUrl: '' })
  const [keyInput, setKeyInput] = useState('')
  const [baseInput, setBaseInput] = useState('https://funpay.com')
  const [nodes, setNodes] = useState([])
  const [newNode, setNewNode] = useState('')
  const [activeNode, setActiveNode] = useState(null)
  const [messages, setMessages] = useState([])
  const [messageText, setMessageText] = useState('')
  const [lots, setLots] = useState([])
  const [loadingLots, setLoadingLots] = useState(false)
  const [loadingMsgs, setLoadingMsgs] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const statusLabel = useMemo(() => {
    if (error) return { text: 'Error', tone: 'danger' }
    if (session.polling) return { text: `Active · ${session.userId || ''}`, tone: 'success' }
    return { text: 'Idle', tone: 'muted' }
  }, [session, error])

  useEffect(() => {
    loadSession()
    loadNodes()
  }, [])

  useEffect(() => {
    if (!activeNode) return
    let timer = setInterval(() => refreshMessages(activeNode, false), 4000)
    return () => clearInterval(timer)
  }, [activeNode])

  const loadSession = async () => {
    try {
      const data = await api('/api/session')
      setSession(data)
      if (data.baseUrl) setBaseInput(data.baseUrl)
      setError('')
    } catch (e) {
      setError(e.message)
    }
  }

  const startSession = async () => {
    if (!keyInput.trim()) {
      setError('Enter a Golden Key')
      return
    }
    setBusy(true)
    try {
      const data = await api('/api/session', {
        method: 'POST',
        body: JSON.stringify({ golden_key: keyInput.trim(), base_url: baseInput.trim() }),
      })
      setSession({ polling: true, userId: data.userId, baseUrl: data.baseUrl })
      setError('')
      await loadNodes()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const loadNodes = async () => {
    try {
      const data = await api('/api/nodes')
      setNodes(data)
    } catch (e) {
      setError(e.message)
    }
  }

  const addNode = async () => {
    if (!newNode.trim()) return
    try {
      await api('/api/nodes', { method: 'POST', body: JSON.stringify({ node: newNode.trim() }) })
      setNewNode('')
      loadNodes()
    } catch (e) {
      setError(e.message)
    }
  }

  const selectNode = async (nodeId) => {
    setActiveNode(nodeId)
    await refreshMessages(nodeId, true)
  }

  const refreshMessages = async (nodeId, withLoading = true) => {
    if (!nodeId) return
    setLoadingMsgs(withLoading)
    try {
      const msgs = await api(`/api/messages?node=${encodeURIComponent(nodeId)}&limit=100`)
      setMessages(msgs)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoadingMsgs(false)
    }
  }

  const sendMessage = async () => {
    if (!activeNode || !messageText.trim()) return
    try {
      await api('/api/messages/send', {
        method: 'POST',
        body: JSON.stringify({ node: activeNode, message: messageText }),
      })
      setMessageText('')
      refreshMessages(activeNode, false)
    } catch (e) {
      setError(e.message)
    }
  }

  const loadLots = async () => {
    setLoadingLots(true)
    try {
      const data = await api('/api/lots')
      setLots(data || [])
      setError('')
    } catch (e) {
      setError(e.message)
    } finally {
      setLoadingLots(false)
    }
  }

  const updatePrice = async (node, offer, price) => {
    try {
      await api('/api/lots/price', {
        method: 'POST',
        body: JSON.stringify({ node, offer, price: Number(price) }),
      })
      await loadLots()
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div className="page">
      <header className="topbar">
        <div>
          <p className="eyebrow">Funpay Control</p>
          <h1>Inbox + Lots Dashboard</h1>
        </div>
        <div className={`pill ${statusLabel.tone}`}>{statusLabel.text}</div>
      </header>

      <div className="grid">
        <section className="panel">
          <div className="panel-header">
            <h2>Session</h2>
            <p className="sub">Golden Key is only stored in this session (memory).</p>
          </div>
          <label>Golden Key</label>
          <input
            value={keyInput}
            onChange={(e) => setKeyInput(e.target.value)}
            placeholder="Paste your golden_key cookie"
          />
          <label>Base URL</label>
          <input value={baseInput} onChange={(e) => setBaseInput(e.target.value)} />
          <button onClick={startSession} disabled={busy}>
            {busy ? 'Starting…' : 'Start Session'}
          </button>
          {error && <div className="alert">{error}</div>}
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Chat Nodes</h2>
            <div className="sub">Add user IDs to monitor + reply.</div>
          </div>
          <div className="row">
            <input
              value={newNode}
              onChange={(e) => setNewNode(e.target.value)}
              placeholder="User ID"
            />
            <button onClick={addNode}>Add</button>
          </div>
          <div className="chips">
            {nodes.length === 0 && <div className="muted">No nodes yet.</div>}
            {nodes.map((n) => (
              <button
                key={n.id}
                className={`chip ${activeNode === n.id ? 'active' : ''}`}
                onClick={() => selectNode(n.id)}
              >
                <span>{n.id}</span>
                <small className="muted">last {n.last_id ?? 0}</small>
              </button>
            ))}
          </div>
        </section>
      </div>

      <div className="grid two">
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Messages {activeNode ? `· ${activeNode}` : ''}</h2>
              <div className="sub">{loadingMsgs ? 'Loading…' : 'Auto-refreshing every 4s'}</div>
            </div>
            <button onClick={() => activeNode && refreshMessages(activeNode, true)}>
              Refresh
            </button>
          </div>
          <div className="messages">
            {!activeNode && <div className="muted">Select a node to view messages.</div>}
            {activeNode && messages.length === 0 && !loadingMsgs && (
              <div className="muted">No messages yet.</div>
            )}
            {messages.map((m) => (
              <article key={m.id} className="msg">
                <div className="msg-head">
                  <div>{m.username || 'Unknown'}</div>
                  <div className="muted">#{m.id}{m.created_at ? ` · ${m.created_at}` : ''}</div>
                </div>
                <p>{m.body}</p>
              </article>
            ))}
          </div>
          <div className="row">
            <textarea
              value={messageText}
              onChange={(e) => setMessageText(e.target.value)}
              rows={2}
              placeholder="Reply…"
            />
            <button onClick={sendMessage} disabled={!activeNode}>
              Send
            </button>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Lots</h2>
              <div className="sub">View your offers and adjust prices.</div>
            </div>
            <button onClick={loadLots} disabled={loadingLots}>
              {loadingLots ? 'Loading…' : 'Refresh'}
            </button>
          </div>
          <div className="lots">
            {lots.length === 0 && !loadingLots && (
              <div className="muted">No lots loaded yet. Click Refresh.</div>
            )}
            {lots.map((group) => (
              <div key={group.node} className="lot-group">
                <div className="lot-head">
                  <div>
                    <div className="muted">{group.node}</div>
                    <h3>{group.group_name}</h3>
                  </div>
                </div>
                <div className="lot-list">
                  {group.offers.map((o, idx) => (
                    <LotRow
                      key={`${group.node}-${o.id || idx}`}
                      groupNode={group.node}
                      offer={o}
                      onUpdate={updatePrice}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}

function LotRow({ groupNode, offer, onUpdate }) {
  const [price, setPrice] = useState(offer.price)
  return (
    <div className="lot-row">
      <div>
        <div className="muted">#{offer.id || 'unknown'}</div>
        <strong>{offer.name}</strong>
      </div>
      <div className="row">
        <input
          value={price}
          onChange={(e) => setPrice(e.target.value)}
          style={{ minWidth: 80 }}
        />
        <button onClick={() => onUpdate(groupNode, offer.id, price)}>Save</button>
      </div>
    </div>
  )
}
