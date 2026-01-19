import { useEffect, useMemo, useState } from 'react'
import {
  AppShell,
  Avatar,
  Badge,
  Box,
  Button,
  Divider,
  Group,
  Paper,
  PasswordInput,
  ScrollArea,
  Stack,
  Text,
  Textarea,
  TextInput,
} from '@mantine/core'
import { useMediaQuery } from '@mantine/hooks'

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
  const [dialogs, setDialogs] = useState([])
  const [activeChatNode, setActiveChatNode] = useState(null)
  const [activeNode, setActiveNode] = useState(null)
  const [messages, setMessages] = useState([])
  const [messageText, setMessageText] = useState('')
  const [loadingMsgs, setLoadingMsgs] = useState(false)
  const [lots, setLots] = useState([])
  const [loadingLots, setLoadingLots] = useState(false)
  const [error, setError] = useState('')
  const [orders, setOrders] = useState([])
  const [loadingOrders, setLoadingOrders] = useState(false)
  const [accounts, setAccounts] = useState([])
  const [accountForm, setAccountForm] = useState({
    label: '',
    username: '',
    password: '',
    guard: '',
  })

  const isNarrow = useMediaQuery('(max-width: 900px)')
  const statusLabel = useMemo(() => {
    if (error) return { text: 'Error', tone: 'danger' }
    if (session.polling) {
      return {
        text: session.userId ? `Active - ${session.userId}` : 'Active',
        tone: 'success',
      }
    }
    return { text: 'Idle', tone: 'muted' }
  }, [session, error])

  useEffect(() => {
    loadSession()
    loadDialogs()
    loadLots()
    loadOrders()
    const dialogsTimer = setInterval(loadDialogs, 30000)
    const ordersTimer = setInterval(loadOrders, 30000)
    return () => {
      clearInterval(dialogsTimer)
      clearInterval(ordersTimer)
    }
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

  const loadDialogs = async () => {
    try {
      const data = await api('/api/dialogs?resolve=1&resolve_limit=200')
      setDialogs(data)
    } catch (e) {
      setError(e.message)
    }
  }

  const loadLots = async () => {
    setLoadingLots(true)
    try {
      const data = await api('/api/lots')
      setLots(Array.isArray(data) ? data : [])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoadingLots(false)
    }
  }

  const loadOrders = async () => {
    setLoadingOrders(true)
    try {
      const data = await api('/api/orders')
      setOrders(Array.isArray(data) ? data : [])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoadingOrders(false)
    }
  }

  const syncMessages = async (nodeId, chatNode, limit = 150) => {
    if (!nodeId && !chatNode) return null
    try {
      const data = await api('/api/messages/sync', {
        method: 'POST',
        body: JSON.stringify({ node: nodeId || null, chat_node: chatNode || null, limit }),
      })
      return data
    } catch (e) {
      setError(e.message)
      return null
    }
  }

  const selectDialog = async (dialog) => {
    setMessages([])
    setLoadingMsgs(true)
    setActiveChatNode(dialog.node_id)
    setActiveNode(dialog.user_id || null)

    let resolvedUserId = dialog.user_id
    const sync = await syncMessages(resolvedUserId, dialog.node_id, 150)
    resolvedUserId = sync?.user_id || resolvedUserId || null
    if (!resolvedUserId) {
      setLoadingMsgs(false)
      return
    }
    setActiveNode(resolvedUserId)
    await refreshMessages(resolvedUserId, true)
  }

  const refreshMessages = async (nodeId, withLoading = true) => {
    if (!nodeId) return
    setLoadingMsgs(withLoading)
    try {
      const msgs = await api(`/api/messages?node=${encodeURIComponent(nodeId)}&limit=150`)
      setMessages(Array.isArray(msgs) ? msgs.slice().reverse() : [])
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

  const addAccount = () => {
    if (!accountForm.username.trim()) return
    const newAccount = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      label: accountForm.label.trim() || accountForm.username.trim(),
      username: accountForm.username.trim(),
      status: 'Not connected',
    }
    setAccounts((prev) => [newAccount, ...prev])
    setAccountForm({ label: '', username: '', password: '', guard: '' })
  }

  const activeDialog = dialogs.find((d) => d.node_id === activeChatNode)
  const activeName = activeDialog?.name || activeDialog?.user_id || activeDialog?.node_id || 'Dialog'

  const badgeColor =
    statusLabel.tone === 'success' ? 'green' : statusLabel.tone === 'danger' ? 'red' : 'gray'

  return (
    <AppShell
      padding="md"
      style={{ height: '100dvh' }}
      styles={{
        main: {
          background: '#15181c',
          height: '100dvh',
        },
      }}
    >
      <AppShell.Main>
        <Box
          style={{
            height: '100%',
            display: 'grid',
            gridTemplateColumns: isNarrow
              ? '1fr'
              : '280px minmax(700px, 1fr) minmax(240px, 280px)',
            gridTemplateRows: isNarrow ? 'minmax(0, 1fr) minmax(0, 1fr)' : '1fr',
            gap: 16,
            maxWidth: 1400,
            margin: '0 auto',
          }}
        >
          <Paper withBorder radius="md" p="md" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <Group justify="space-between" mb="xs">
              <Text fw={700}>Messages</Text>
              <Badge color={badgeColor} variant="light">
                {statusLabel.text}
              </Badge>
            </Group>
            <ScrollArea style={{ flex: 1, minHeight: 0 }} offsetScrollbars scrollbarSize={8}>
              <Stack gap="xs">
                {dialogs.length === 0 && (
                  <Text size="sm" c="dimmed">
                    No dialogs yet.
                  </Text>
                )}
                {dialogs.map((d) => {
                  const name = d.name || d.user_id || d.node_id
                  const preview = d.preview || `#${d.node_id}`
                  const letter = name ? name.slice(0, 1).toUpperCase() : '?'
                  const isActive = activeChatNode === d.node_id
                  const avatarSrc = d.avatar
                    ? `/api/avatar?url=${encodeURIComponent(d.avatar)}`
                    : null
                  return (
                    <Paper
                      key={d.node_id}
                      withBorder
                      radius="md"
                      p="sm"
                      onClick={() => selectDialog(d)}
                      style={{
                        cursor: 'pointer',
                        background: isActive ? '#2f3642' : 'transparent',
                      }}
                    >
                      <Group align="flex-start" gap="sm" wrap="nowrap">
                        <Avatar
                          src={avatarSrc}
                          radius="xl"
                          color="gray"
                          imageProps={{ loading: 'eager' }}
                        >
                          {letter}
                        </Avatar>
                        <Box style={{ flex: 1, minWidth: 0 }}>
                          <Group justify="space-between" gap="xs" wrap="nowrap">
                            <Text fw={600} size="sm" truncate>
                              {name}
                            </Text>
                            <Text size="xs" c="dimmed" style={{ whiteSpace: 'nowrap' }}>
                              {d.time || ''}
                            </Text>
                          </Group>
                          <Text size="xs" c="dimmed" truncate>
                            {preview}
                          </Text>
                        </Box>
                      </Group>
                    </Paper>
                  )
                })}
              </Stack>
            </ScrollArea>
          </Paper>

          <Paper withBorder radius="md" p="md" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <Group justify="space-between" mb="xs">
              <Text fw={700}>{activeName}</Text>
              <Text size="xs" c="dimmed">
                {loadingMsgs ? 'Loading...' : 'Auto-refresh 4s'}
              </Text>
            </Group>
            <ScrollArea style={{ flex: 1, minHeight: 0 }} offsetScrollbars scrollbarSize={8}>
              <Stack gap="sm">
                {!activeChatNode && (
                  <Text size="sm" c="dimmed">
                    Select a dialog on the left.
                  </Text>
                )}
                {activeChatNode && activeNode && messages.length === 0 && !loadingMsgs && (
                  <Text size="sm" c="dimmed">
                    No messages yet.
                  </Text>
                )}
                {messages.map((m) => (
                  <Paper key={m.id} withBorder radius="md" p="sm" style={{ background: '#242a33' }}>
                    <Group justify="space-between" gap="xs" mb={4}>
                      <Text size="xs" fw={600}>
                        {m.username || 'Unknown'}
                      </Text>
                      <Text size="xs" c="dimmed">
                        #{m.id}
                        {m.created_at ? ` at ${m.created_at}` : ''}
                      </Text>
                    </Group>
                    <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
                      {m.body}
                    </Text>
                  </Paper>
                ))}
              </Stack>
            </ScrollArea>
            <Group align="stretch" mt="sm" wrap="nowrap">
              <Textarea
                value={messageText}
                onChange={(e) => setMessageText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    sendMessage()
                  }
                }}
                placeholder="Write a message..."
                autosize
                minRows={3}
                maxRows={5}
                style={{ flex: 1 }}
              />
              <Button
                onClick={sendMessage}
                disabled={!activeNode}
                h="100%"
                style={{ alignSelf: 'stretch' }}
                styles={{ root: { height: '100%' } }}
              >
                Send
              </Button>
            </Group>
          </Paper>

          {!isNarrow && (
            <Stack style={{ minHeight: 0 }}>
              <Paper withBorder radius="md" p="md" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                <Group justify="space-between" mb="xs">
                  <Text fw={700}>Accounts</Text>
                  <Badge color="gray" variant="light">
                    UI only
                  </Badge>
                </Group>
                <Stack gap="xs">
                  <TextInput
                    label="Label"
                    placeholder="Main account"
                    value={accountForm.label}
                    onChange={(e) => setAccountForm((prev) => ({ ...prev, label: e.target.value }))}
                  />
                  <TextInput
                    label="Username"
                    placeholder="steam_login"
                    value={accountForm.username}
                    onChange={(e) => setAccountForm((prev) => ({ ...prev, username: e.target.value }))}
                  />
                  <PasswordInput
                    label="Password"
                    placeholder="Not stored"
                    value={accountForm.password}
                    onChange={(e) => setAccountForm((prev) => ({ ...prev, password: e.target.value }))}
                  />
                  <TextInput
                    label="Steam Guard code"
                    placeholder="12345"
                    value={accountForm.guard}
                    onChange={(e) => setAccountForm((prev) => ({ ...prev, guard: e.target.value }))}
                  />
                  <Button onClick={addAccount} disabled={!accountForm.username.trim()}>
                    Add account
                  </Button>
                  <Text size="xs" c="dimmed">
                    Credentials are not saved. Hook a backend to actually log in.
                  </Text>
                </Stack>
                <Divider my="sm" />
                <ScrollArea style={{ flex: 1, minHeight: 0 }} offsetScrollbars scrollbarSize={8}>
                  <Stack gap="sm">
                    {accounts.length === 0 && (
                      <Text size="sm" c="dimmed">
                        No accounts yet.
                      </Text>
                    )}
                    {accounts.map((acc) => (
                      <Paper key={acc.id} withBorder radius="md" p="sm">
                        <Group justify="space-between" wrap="nowrap">
                          <Box style={{ minWidth: 0 }}>
                            <Text size="sm" fw={600} truncate>
                              {acc.label}
                            </Text>
                            <Text size="xs" c="dimmed" truncate>
                              {acc.username}
                            </Text>
                          </Box>
                          <Badge color="gray" variant="light">
                            {acc.status}
                          </Badge>
                        </Group>
                      </Paper>
                    ))}
                  </Stack>
                </ScrollArea>
              </Paper>

              <Paper withBorder radius="md" p="md" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                <Group justify="space-between" mb="xs">
                  <Text fw={700}>Active orders</Text>
                  <Button variant="subtle" size="xs" onClick={loadOrders} loading={loadingOrders}>
                    Refresh
                  </Button>
                </Group>
                <ScrollArea style={{ flex: 1, minHeight: 0 }} offsetScrollbars scrollbarSize={8}>
                  <Stack gap="sm">
                    {orders.length === 0 && (
                      <Text size="sm" c="dimmed">
                        {loadingOrders ? 'Loading orders...' : 'No active orders.'}
                      </Text>
                    )}
                    {orders.map((order) => (
                      <Paper key={order.order_id} withBorder radius="md" p="sm">
                        <Group justify="space-between" gap="xs" wrap="nowrap">
                          <Text size="sm" fw={600} truncate>
                            #{order.order_id}
                          </Text>
                          <Group gap="xs" wrap="nowrap">
                            {order.is_new && (
                              <Badge color="green" variant="light">
                                New
                              </Badge>
                            )}
                            <Badge color="gray" variant="light">
                              {order.status || 'Unknown'}
                            </Badge>
                          </Group>
                        </Group>
                        <Text size="xs" c="dimmed" truncate>
                          {order.product || 'Order'}{order.amount ? ` x${order.amount}` : ''}
                        </Text>
                        <Text size="xs" c="dimmed" truncate>
                          {order.date || ''}{order.user_id ? ` • User ${order.user_id}` : ''}
                        </Text>
                      </Paper>
                    ))}
                  </Stack>
                </ScrollArea>
              </Paper>

              <Paper withBorder radius="md" p="md" style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                <Group justify="space-between" mb="xs">
                  <Text fw={700}>Active lots</Text>
                  <Button variant="subtle" size="xs" onClick={loadLots} loading={loadingLots}>
                    Refresh
                  </Button>
                </Group>
                <ScrollArea style={{ flex: 1, minHeight: 0 }} offsetScrollbars scrollbarSize={8}>
                  <Stack gap="sm">
                    {lots.length === 0 && (
                      <Text size="sm" c="dimmed">
                        {loadingLots ? 'Loading lots...' : 'No lots available.'}
                      </Text>
                    )}
                    {lots.map((group) => (
                      <Paper key={group.node || group.group_name} withBorder radius="md" p="sm">
                        <Text fw={600} size="sm" mb={4}>
                          {group.group_name || 'Group'}
                        </Text>
                        <Stack gap={6}>
                          {(group.offers || []).map((offer) => (
                            <Group
                              key={`${group.node || group.group_name}-${offer.id || offer.name}`}
                              justify="space-between"
                              wrap="nowrap"
                            >
                              <Text size="xs" truncate>
                                {offer.name || 'Offer'}
                              </Text>
                              <Text size="xs" c="dimmed" style={{ whiteSpace: 'nowrap' }}>
                                {offer.price || ''}
                              </Text>
                            </Group>
                          ))}
                        </Stack>
                      </Paper>
                    ))}
                  </Stack>
                </ScrollArea>
              </Paper>
            </Stack>
          )}
        </Box>
      </AppShell.Main>
    </AppShell>
  )
}
