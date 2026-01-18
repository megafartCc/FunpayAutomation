import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
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
  const [keyInput, setKeyInput] = useState('')
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
    if (error) return { text: 'Ошибка', tone: 'danger' }
    if (session.polling) return { text: `В сети · ${session.userId || ''}`, tone: 'success' }
    return { text: 'Ожидание', tone: 'muted' }
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

  const startSession = async () => {
    if (!keyInput.trim()) {
      setError('Введите Golden Key')
      return
    }
    setBusy(true)
    try {
      const data = await api('/api/session', {
        method: 'POST',
        body: JSON.stringify({ golden_key: keyInput.trim() }),
      })
      setSession({ polling: true, userId: data.userId, baseUrl: data.baseUrl || 'https://funpay.com' })
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
    <div className="page dark">
      <header className="topbar">
        <div>
          <h1 className="brand">FunPay Panel</h1>
          <p className="sub">Сообщения и лоты</p>
        </div>
        <div className={`pill ${statusLabel.tone}`}>{statusLabel.text}</div>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <motion.section className="panel flat" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <div className="panel-header">
              <div>
                <h2>Сессия</h2>
                <p className="sub">Ключ хранится только в памяти.</p>
              </div>
              <div className="pill muted small">funpay.com</div>
            </div>
            <label>Golden Key</label>
            <input
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              placeholder="Вставьте golden_key cookie"
            />
            <button onClick={startSession} disabled={busy}>
              {busy ? 'Запуск…' : 'Запустить'}
            </button>
            {error && <div className="alert">{error}</div>}
          </motion.section>

          <motion.section className="panel flat" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <div className="panel-header">
              <div>
                <h2>Диалоги</h2>
                <div className="sub">Добавьте пользователя.</div>
              </div>
              <span className="pill muted small">{nodes.length}</span>
            </div>
            <div className="row">
              <input
                value={newNode}
                onChange={(e) => setNewNode(e.target.value)}
                placeholder="User ID"
              />
              <button onClick={addNode}>Добавить</button>
            </div>
            <div className="contact-list tall">
              {nodes.length === 0 && <div className="muted">Нет диалогов.</div>}
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
          </motion.section>
        </aside>

        <main className="main wide">
          <motion.section className="panel chat-panel" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <div className="panel-header">
              <div>
                <h2>Сообщения {activeNode ? `· ${activeNode}` : ''}</h2>
                <div className="sub">{loadingMsgs ? 'Загрузка…' : 'Автообновление каждые 4с'}</div>
              </div>
              <button onClick={() => activeNode && refreshMessages(activeNode, true)}>Обновить</button>
            </div>
            <div className="messages">
              {!activeNode && <div className="muted">Выберите диалог слева.</div>}
              {activeNode && messages.length === 0 && !loadingMsgs && (
                <div className="muted">Нет сообщений.</div>
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
                placeholder="Написать..."
              />
              <button onClick={sendMessage} disabled={!activeNode}>
                Отправить
              </button>
            </div>
          </motion.section>

          <motion.section className="panel lots-panel" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <div className="panel-header">
              <div>
                <h2>Лоты</h2>
                <div className="sub">Ваши предложения, цены.</div>
              </div>
              <button onClick={loadLots} disabled={loadingLots}>
                {loadingLots ? 'Загрузка…' : 'Обновить'}
              </button>
            </div>
            <div className="lots">
              {lots.length === 0 && !loadingLots && (
                <div className="muted">Пока пусто. Нажмите «Обновить».</div>
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
          </motion.section>
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
        <button onClick={() => onUpdate(groupNode, offer.id, price)}>Сохранить</button>
      </div>
    </div>
  )
}
