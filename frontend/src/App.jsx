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
  const [session, setSession] = useState({ polling: false, userId: null, baseUrl: 'https://funpay.com' })
  const [nodes, setNodes] = useState([])
  const [newNode, setNewNode] = useState('')
  const [activeNode, setActiveNode] = useState(null)
  const [messages, setMessages] = useState([])
  const [messageText, setMessageText] = useState('')
  const [lots, setLots] = useState([])
  const [loadingLots, setLoadingLots] = useState(false)
  const [loadingMsgs, setLoadingMsgs] = useState(false)
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
    const timer = setInterval(() => refreshMessages(activeNode, false), 4000)
    return () => clearInterval(timer)
  }, [activeNode])

  const loadSession = async () => {
    try {
      const data = await api('/api/session')
      setSession(data)
      setError('')
    } catch (e) {
      setError(e.message)
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
    <div className="page dark">
      <header className="topbar">
        <div>
          <h1 className="brand">FunPay Panel</h1>
          <p className="sub">Chats and lots</p>
        </div>
        <div className={`pill ${statusLabel.tone}`}>{statusLabel.text}</div>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <div className="panel flat">
            <div className="panel-header">
              <div>
                <h2>Session</h2>
                <p className="sub">Using server Golden Key (env).</p>
              </div>
              <div className="pill muted small">funpay.com</div>
            </div>
            <div className="pill success small">{statusLabel.text}</div>
            {error && <div className="alert">{error}</div>}
          </div>

          <div className="panel flat">
            <div className="panel-header">
              <div>
                <h2>Dialogs</h2>
                <div className="sub">Add user IDs.</div>
              </div>
              <span className="pill muted small">{nodes.length}</span>
            </div>
            <div className="row">
              <input
                value={newNode}
                onChange={(e) => setNewNode(e.target.value)}
                placeholder="User ID"
              />
              <button onClick={addNode}>Add</button>
            </div>
            <div className="contact-list tall">
              {nodes.length === 0 && <div className="muted">No dialogs yet.</div>}
              {nodes.map((n) => (
                <div
                  key={n.id}
                  className={`contact-item ${activeNode === n.id ? 'active' : ''}`}
                  onClick={() => selectNode(n.id)}
                >
                  <div className="contact-title">{n.last_username || n.id}</div>
                  <div className="contact-meta">
                    {n.last_body ? n.last_body.slice(0, 60) : `#${n.id}`} · {n.last_created_at || '—'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </aside>

        <main className="main wide">
          <div className="panel chat-panel">
            <div className="panel-header">
              <div>
                <h2>Messages {activeNode ? `· ${activeNode}` : ''}</h2>
                <div className="sub">{loadingMsgs ? 'Loading…' : 'Auto-refresh every 4s'}</div>
              </div>
              <button onClick={() => activeNode && refreshMessages(activeNode, true)}>Refresh</button>
            </div>
            <div className="messages">
              {!activeNode && <div className="muted">Select a dialog on the left.</div>}
              {activeNode && messages.length === 0 && !loadingMsgs && (
                <div className="muted">No messages yet.</div>
              )}
              {messages.map((m) => (
                <article key={m.id} className="msg">
                  <div className="msg-head">
                    <div className="msg-author">{m.username || 'Unknown'}</div>
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
                placeholder="Write a reply..."
              />
              <button onClick={sendMessage} disabled={!activeNode}>
                Send
              </button>
            </div>
          </div>

          <div className="panel lots-panel">
            <div className="panel-header">
              <div>
                <h2>Lots</h2>
                <div className="sub">Your offers and prices.</div>
              </div>
              <button onClick={loadLots} disabled={loadingLots}>
                {loadingLots ? 'Loading…' : 'Refresh'}
              </button>
            </div>
            <div className="lots">
              {lots.length === 0 && !loadingLots && (
                <div className="muted">Empty. Click “Refresh”.</div>
              )}
              {lots.map((group) => (
                <div key={group.node} className="lot-group">
                  <div className="lot-head">
                    <div className="muted">{group.node}</div>
                    <h3>{group.group_name}</h3>
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
          </div>
        </main>
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
          style={{ minWidth: 90 }}
        />
        <button onClick={() => onUpdate(groupNode, offer.id, price)}>Save</button>
      </div>
    </div>
  )
}
