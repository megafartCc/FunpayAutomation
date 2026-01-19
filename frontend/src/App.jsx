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
  const [session, setSession] = useState({ polling: false, userId: null })
  const [nodes, setNodes] = useState([])
  const [activeNode, setActiveNode] = useState(null)
  const [messages, setMessages] = useState([])
  const [messageText, setMessageText] = useState('')
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

  const activeDialog = nodes.find((n) => n.id === activeNode)
  const activeName = activeDialog?.partner_name || activeDialog?.last_username || activeDialog?.id || 'Dialog'

  return (
    <div className="page">
      <header className="logo-bar">
        <img src="/logo-funpay.svg" alt="FunPay" className="logo" />
      </header>

      <div className="content">
        <aside className="dialogs">
          <div className="dialogs-header">
            <div className="title">Messages</div>
            <div className={`pill ${statusLabel.tone}`}>{statusLabel.text}</div>
          </div>
          <div className="dialog-list">
            {nodes.length === 0 && <div className="muted">No dialogs yet.</div>}
            {nodes.map((n) => {
              const name = n.partner_name || n.last_username || n.id
              const preview = n.last_body ? n.last_body.slice(0, 60) : `#${n.id}`
              return (
                <div
                  key={n.id}
                  className={`dialog-item ${activeNode === n.id ? 'active' : ''}`}
                  onClick={() => selectNode(n.id)}
                >
                  <Avatar name={name} src={n.partner_avatar} />
                  <div className="dialog-body">
                    <div className="dialog-top">
                      <div className="dialog-name">{name}</div>
                      <div className="dialog-time">{n.last_created_at || ''}</div>
                    </div>
                    <div className="dialog-preview">{preview}</div>
                  </div>
                </div>
              )
            })}
          </div>
        </aside>

        <section className="chat">
          <div className="chat-header">
            <div className="chat-title">{activeName}</div>
            <div className="chat-sub">{loadingMsgs ? 'Loading…' : 'Auto-refresh 4s'}</div>
          </div>
          <div className="chat-body">
            {!activeNode && <div className="muted">Select a dialog on the left.</div>}
            {activeNode && messages.length === 0 && !loadingMsgs && (
              <div className="muted">No messages yet.</div>
            )}
            {messages.map((m) => (
              <div key={m.id} className="message">
                <div className="message-meta">
                  <span className="message-author">{m.username || 'Unknown'}</span>
                  <span className="message-time">#{m.id}{m.created_at ? ` · ${m.created_at}` : ''}</span>
                </div>
                <div className="message-text">{m.body}</div>
              </div>
            ))}
          </div>
          <div className="chat-input">
            <textarea
              value={messageText}
              onChange={(e) => setMessageText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  sendMessage()
                }
              }}
              placeholder="Write a message..."
              rows={2}
            />
            <button onClick={sendMessage} disabled={!activeNode}>
              Send
            </button>
          </div>
        </section>
      </div>
    </div>
  )
}

function Avatar({ name, src }) {
  if (src) {
    return <img className="avatar" src={src} alt={name || 'avatar'} />
  }
  const letter = name ? name.slice(0, 1).toUpperCase() : '?'
  return <div className="avatar placeholder">{letter}</div>
}
