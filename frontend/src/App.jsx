import { useEffect, useMemo, useState } from 'react'
import {
  Alert,
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
  const [accounts, setAccounts] = useState([])
  const [accountForm, setAccountForm] = useState({
    label: '',
    username: '',
    password: '',
    steamId: '',
    guard: '',
  })
  const [accountGuards, setAccountGuards] = useState({})

  const isNarrow = useMediaQuery('(max-width: 900px)')
  const isWide = useMediaQuery('(min-width: 1400px)')
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
    loadAccounts()
    const dialogsTimer = setInterval(loadDialogs, 30000)
    return () => {
      clearInterval(dialogsTimer)
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

  const loadAccounts = async () => {
    try {
      const data = await api('/api/accounts')
      setAccounts(Array.isArray(data) ? data : [])
    } catch (e) {
      setError(e.message)
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

  const addAccount = async () => {
    if (!accountForm.username.trim() || !accountForm.password.trim()) return
    try {
      await api('/api/accounts', {
        method: 'POST',
        body: JSON.stringify({
          label: accountForm.label.trim() || null,
          username: accountForm.username.trim(),
          password: accountForm.password.trim(),
          steam_id: accountForm.steamId.trim() || null,
          login_status: 'idle',
          twofa_otp: accountForm.guard.trim() || null,
        }),
      })
      setAccountForm({ label: '', username: '', password: '', steamId: '', guard: '' })
      loadAccounts()
    } catch (e) {
      setError(e.message)
    }
  }

  const loginAccount = async (accountId) => {
    try {
      await api(`/api/accounts/${accountId}/login`, {
        method: 'POST',
        body: JSON.stringify({ guard_code: accountGuards[accountId] || '' }),
      })
      setAccountGuards((prev) => ({ ...prev, [accountId]: '' }))
      loadAccounts()
    } catch (e) {
      setError(e.message)
    }
  }

  const logoutAccount = async (accountId) => {
    try {
      await api(`/api/accounts/${accountId}/logout`, { method: 'POST' })
      loadAccounts()
    } catch (e) {
      setError(e.message)
    }
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
        {error && (
          <Alert color="red" mb="md" onClose={() => setError('')} withCloseButton>
            {error}
          </Alert>
        )}
        <Box
          style={{
            height: '100%',
            display: 'grid',
            gridTemplateColumns: isNarrow
              ? '1fr'
              : '280px minmax(700px, 1fr) minmax(560px, 720px)',
            gridTemplateRows: isNarrow ? 'minmax(0, 1fr) minmax(0, 1fr)' : '1fr',
            gap: 16,
            maxWidth: isWide ? 1800 : 1600,
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
            <Box
              style={{
                display: 'grid',
                gridTemplateColumns: 'minmax(260px, 1fr) minmax(260px, 1fr)',
                gap: 16,
                minHeight: 0,
              }}
            >
              <Stack style={{ minHeight: 0 }}>
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
                    label="SteamID"
                    placeholder="7656119..."
                    value={accountForm.steamId}
                    onChange={(e) => setAccountForm((prev) => ({ ...prev, steamId: e.target.value }))}
                  />
                  <TextInput
                    label="2FA OTP"
                    placeholder="12345"
                    value={accountForm.guard}
                    onChange={(e) => setAccountForm((prev) => ({ ...prev, guard: e.target.value }))}
                  />
                  <Button
                    onClick={addAccount}
                    disabled={!accountForm.username.trim() || !accountForm.password.trim()}
                  >
                    Add account
                  </Button>
                  <Text size="xs" c="dimmed">
                    Stored in database. Keep secrets secure.
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
                        <Group justify="space-between" wrap="nowrap" align="flex-start">
                          <Box style={{ minWidth: 0 }}>
                            <Text size="sm" fw={600} truncate>
                              {acc.label || acc.username}
                            </Text>
                            <Text size="xs" c="dimmed" truncate>
                              {acc.steam_id || 'SteamID not set'}
                            </Text>
                            <Group gap="xs" mt={6} style={{ width: '100%' }}>
                              <TextInput
                                size="xs"
                                placeholder={
                                  acc.login_status === 'guard:email'
                                    ? 'Enter email code from Steam'
                                    : acc.login_status === 'guard:twofactor'
                                    ? 'Enter 2FA code'
                                    : 'Guard code (if needed)'
                                }
                                value={accountGuards[acc.id] || ''}
                                onChange={(e) =>
                                  setAccountGuards((prev) => ({ ...prev, [acc.id]: e.target.value }))
                                }
                                style={{ flex: 1 }}
                                disabled={acc.login_status === 'online'}
                              />
                              <Button 
                                size="xs" 
                                onClick={() => loginAccount(acc.id)}
                                disabled={acc.login_status === 'online'}
                              >
                                {acc.login_status === 'guard:email' || acc.login_status === 'guard:twofactor'
                                  ? 'Retry with Code'
                                  : 'Login'}
                              </Button>
                              <Button 
                                size="xs" 
                                variant="subtle" 
                                onClick={() => logoutAccount(acc.id)}
                                disabled={acc.login_status !== 'online'}
                              >
                                Logout
                              </Button>
                            </Group>
                            {acc.login_status === 'guard:email' && (
                              <Text size="xs" c="yellow" mt={4}>
                                ⚠️ Check your email for the Steam verification code
                              </Text>
                            )}
                            {acc.login_status === 'guard:twofactor' && (
                              <Text size="xs" c="yellow" mt={4}>
                                ⚠️ Enter your Steam Guard mobile authenticator code
                              </Text>
                            )}
                          </Box>
                          <Badge
                            color={
                              acc.login_status === 'online'
                                ? 'green'
                                : acc.login_status?.startsWith('guard:')
                                ? 'yellow'
                                : acc.login_status?.startsWith('error:')
                                ? 'red'
                                : 'gray'
                            }
                            variant="light"
                          >
                            {acc.login_status === 'guard:email'
                              ? 'Need Email Code'
                              : acc.login_status === 'guard:twofactor'
                              ? 'Need 2FA Code'
                              : acc.login_status || 'idle'}
                          </Badge>
                        </Group>
                      </Paper>
                    ))}
                  </Stack>
                </ScrollArea>
              </Paper>
            </Box>
          )}
        </Box>
      </AppShell.Main>
    </AppShell>
  )
}
