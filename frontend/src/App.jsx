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
  Modal,
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

  const text = await res.text()

  if (!res.ok) {
    let detail
    try {
      const data = JSON.parse(text || '{}')
      detail = data.detail || JSON.stringify(data)
    } catch {
      detail = text || res.statusText
    }
    throw new Error(detail || res.statusText)
  }

  try {
    return JSON.parse(text || '{}')
  } catch {
    return text
  }
}

const getAccountStatusInfo = (status) => {
  if (status === 'online') {
    return { color: 'green', label: 'Online' }
  }
  if (status === 'guard:email') {
    return { color: 'yellow', label: 'Email Code Required' }
  }
  if (status === 'guard:twofactor') {
    return { color: 'yellow', label: '2FA Code Required' }
  }
  if (status && status.startsWith('error:')) {
    const key = status.split(':')[1] || 'unknown'
    const labels = {
      invalid_password: 'Invalid Password',
      invalid_auth_code: 'Invalid Code',
      account_not_found: 'Account Not Found',
      rate_limit: 'Rate Limited',
      init: 'Init Failed',
      connect: 'Connection Failed',
      exception: 'Login Error',
    }
    return { color: 'red', label: labels[key] || 'Error' }
  }
  return { color: 'gray', label: 'Idle' }
}

const getAccountErrorText = (status) => {
  if (status === 'error:invalid_password') {
    return 'Invalid username or password. Please check your credentials.'
  }
  if (status === 'error:invalid_auth_code') {
    return 'The code you entered is incorrect or expired. Please try again.'
  }
  if (status === 'error:rate_limit') {
    return 'Too many login attempts. Please wait a few minutes before trying again.'
  }
  if (status === 'error:account_not_found') {
    return 'Steam account not found. Check your username.'
  }
  return 'An error occurred during login. Please try again.'
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
  const [accounts, setAccounts] = useState([])
  const [accountForm, setAccountForm] = useState({
    label: '',
    username: '',
    password: '',
    steamId: '',
    guard: '',
  })
  const [accountBusy, setAccountBusy] = useState({})
  const [loginCodes, setLoginCodes] = useState({})
  const [passwordModal, setPasswordModal] = useState(null)
  const [error, setError] = useState('')

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

  const loadDialogs = async (retries = 3) => {
    for (let i = 0; i < retries; i++) {
      try {
        const data = await api('/api/dialogs?resolve=1&resolve_limit=200')
        setDialogs(Array.isArray(data) ? data : [])
        return
      } catch (e) {
        console.error(`Failed to load dialogs (attempt ${i + 1}/${retries}):`, e)
        if (i === retries - 1) {
          console.warn('Dialogs failed to load after retries')
        } else {
          await new Promise((resolve) => setTimeout(resolve, 1000 * (i + 1)))
        }
      }
    }
  }

  const loadLots = async (retries = 3) => {
    setLoadingLots(true)
    for (let i = 0; i < retries; i++) {
      try {
        const data = await api('/api/lots')
        setLots(Array.isArray(data) ? data : [])
        setLoadingLots(false)
        return
      } catch (e) {
        console.error(`Failed to load lots (attempt ${i + 1}/${retries}):`, e)
        if (i === retries - 1) {
          setError('Failed to load lots: ' + e.message)
          setLoadingLots(false)
        } else {
          await new Promise((resolve) => setTimeout(resolve, 1000 * (i + 1)))
        }
      }
    }
  }

  const loadAccounts = async (retries = 3) => {
    for (let i = 0; i < retries; i++) {
      try {
        const data = await api('/api/accounts')
        setAccounts(Array.isArray(data) ? data : [])
        return
      } catch (e) {
        console.error(`Failed to load accounts (attempt ${i + 1}/${retries}):`, e)
        if (i === retries - 1) {
          setError('Failed to load accounts: ' + e.message)
        } else {
          await new Promise((resolve) => setTimeout(resolve, 1000 * (i + 1)))
        }
      }
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
      await loadAccounts()
    } catch (e) {
      setError(e.message)
    }
  }

  const loginAccount = async (accountId) => {
    setAccountBusy((prev) => ({ ...prev, [accountId]: true }))
    setError('')
    try {
      const code = loginCodes[accountId] || ''
      await api(`/api/accounts/${accountId}/login`, {
        method: 'POST',
        body: JSON.stringify({ guard_code: code, email_code: code }),
      })
      setLoginCodes((prev) => ({ ...prev, [accountId]: '' }))
      await loadAccounts()
    } catch (e) {
      setError(e.message || 'Login failed')
      await loadAccounts()
    } finally {
      setAccountBusy((prev) => ({ ...prev, [accountId]: false }))
    }
  }

  const logoutAccount = async (accountId) => {
    try {
      await api(`/api/accounts/${accountId}/logout`, { method: 'POST' })
      await loadAccounts()
    } catch (e) {
      setError(e.message)
    }
  }

  const deauthorizeAccount = async (accountId) => {
    try {
      await api(`/api/accounts/${accountId}/deauthorize`, { method: 'POST' })
      setError('')
      await loadAccounts()
    } catch (e) {
      setError(e.message)
    }
  }

  const stopRental = async (accountId) => {
    const confirmed = window.confirm(
      'Are you sure you want to stop the rental? This will change the password and log off the renter.'
    )
    if (!confirmed) return
    try {
      const result = await api(`/api/accounts/${accountId}/stop-rental`, { method: 'POST' })
      let message = result.message || 'Rental stopped successfully.'
      if (result.warnings && result.warnings.length > 0) {
        message += `\n${result.warnings.join('\n')}`
      }
      window.alert(message)
      await loadAccounts()
    } catch (e) {
      setError(e.message)
    }
  }

  const openChangePassword = (accountId) => {
    setPasswordModal({ accountId, newPassword: '' })
  }

  const changePassword = async () => {
    if (!passwordModal?.accountId) return
    if (!passwordModal.newPassword || passwordModal.newPassword.length < 8) {
      setError('Password must be at least 8 characters long.')
      return
    }
    try {
      await api(`/api/accounts/${passwordModal.accountId}/change-password`, {
        method: 'POST',
        body: JSON.stringify({ new_password: passwordModal.newPassword }),
      })
      setError('')
      setPasswordModal(null)
      await loadAccounts()
    } catch (e) {
      setError(e.message)
    }
  }

  const getTwoFactorCode = async (accountId) => {
    try {
      const result = await api(`/api/accounts/${accountId}/code`)
      if (result.code) {
        setLoginCodes((prev) => ({ ...prev, [accountId]: result.code }))
        setError('')
      }
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
                    Live
                  </Badge>
                </Group>
                <Stack gap="xs">
                  <TextInput
                    label="Label"
                    placeholder="Main account"
                    value={accountForm.label}
                    onChange={(e) =>
                      setAccountForm((prev) => ({ ...prev, label: e.target.value }))
                    }
                  />
                  <TextInput
                    label="Username"
                    placeholder="steam_login"
                    value={accountForm.username}
                    onChange={(e) =>
                      setAccountForm((prev) => ({ ...prev, username: e.target.value }))
                    }
                  />
                  <PasswordInput
                    label="Password"
                    placeholder="Not stored"
                    value={accountForm.password}
                    onChange={(e) =>
                      setAccountForm((prev) => ({ ...prev, password: e.target.value }))
                    }
                  />
                  <TextInput
                    label="SteamID"
                    placeholder="7656119..."
                    value={accountForm.steamId}
                    onChange={(e) =>
                      setAccountForm((prev) => ({ ...prev, steamId: e.target.value }))
                    }
                  />
                  <TextInput
                    label="2FA OTP"
                    placeholder="Shared secret or 5-char code"
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
                    {accounts.map((account) => {
                      const busy = accountBusy[account.id] || false
                      const isOnline = account.login_status === 'online'
                      const isGuardEmail = account.login_status === 'guard:email'
                      const isGuardTwoFactor = account.login_status === 'guard:twofactor'
                      const isError = account.login_status?.startsWith('error:')
                      const statusInfo = getAccountStatusInfo(account.login_status)

                      return (
                        <Paper
                          key={account.id}
                          withBorder
                          radius="md"
                          p="md"
                          style={{
                            borderColor: isOnline
                              ? 'var(--mantine-color-green-6)'
                              : isGuardEmail || isGuardTwoFactor
                                ? 'var(--mantine-color-yellow-6)'
                                : isError
                                  ? 'var(--mantine-color-red-6)'
                                  : undefined,
                          }}
                        >
                          <Stack gap="sm">
                            <Group justify="space-between" wrap="nowrap">
                              <Box style={{ flex: 1, minWidth: 0 }}>
                                <Group gap="xs" mb={4}>
                                  <Text size="sm" fw={600} truncate>
                                    {account.label || account.username}
                                  </Text>
                                  {busy && (
                                    <Text size="xs" c="dimmed">
                                      Working...
                                    </Text>
                                  )}
                                </Group>
                                <Text size="xs" c="dimmed" truncate>
                                  {account.steam_id
                                    ? `SteamID: ${account.steam_id}`
                                    : 'SteamID: Not set'}
                                </Text>
                              </Box>
                              <Badge color={statusInfo.color} variant="light" size="sm">
                                {statusInfo.label}
                              </Badge>
                            </Group>

                            {isGuardEmail && (
                              <Alert color="yellow" variant="light" p="xs" radius="sm">
                                <Text size="xs">
                                  Email guard code required. Check your inbox and enter the code below.
                                </Text>
                              </Alert>
                            )}
                            {isGuardTwoFactor && (
                              <Alert color="yellow" variant="light" p="xs" radius="sm">
                                <Text size="xs">
                                  Two-factor code required. Enter your 5-character mobile code below.
                                </Text>
                              </Alert>
                            )}
                            {isError && (
                              <Alert color="red" variant="light" p="xs" radius="sm">
                                <Text size="xs">{getAccountErrorText(account.login_status)}</Text>
                              </Alert>
                            )}
                            {isOnline && (
                              <Alert color="green" variant="light" p="xs" radius="sm">
                                <Text size="xs">Account is online and ready to use.</Text>
                              </Alert>
                            )}

                            {!isOnline && (
                              <Stack gap="xs">
                                <TextInput
                                  size="sm"
                                  placeholder={
                                    isGuardEmail
                                      ? 'Enter 5-character email code (e.g., A1B2C)'
                                      : isGuardTwoFactor
                                        ? 'Enter 5-character 2FA code'
                                        : 'Enter code if required (leave empty for first attempt)'
                                  }
                                  value={loginCodes[account.id] || ''}
                                  onChange={(e) => {
                                    const next = e.target.value
                                      .toUpperCase()
                                      .replace(/[^A-Z0-9]/g, '')
                                      .slice(0, 5)
                                    setLoginCodes((prev) => ({ ...prev, [account.id]: next }))
                                  }}
                                  disabled={busy}
                                  maxLength={5}
                                  style={{ fontFamily: 'monospace', letterSpacing: '2px' }}
                                />
                                <Group gap="xs">
                                  <Button
                                    size="sm"
                                    onClick={() => loginAccount(account.id)}
                                    disabled={busy || isOnline}
                                    loading={busy}
                                    style={{ flex: 1 }}
                                  >
                                    {busy
                                      ? 'Logging in...'
                                      : isGuardEmail || isGuardTwoFactor
                                        ? 'Submit Code'
                                        : 'Login'}
                                  </Button>
                                </Group>
                              </Stack>
                            )}

                            {isOnline && (
                              <Stack gap="xs" mt="sm">
                                <Divider label="Control Panel" labelPosition="center" />
                                <Button
                                  size="md"
                                  variant="filled"
                                  color="red"
                                  onClick={() => stopRental(account.id)}
                                  fullWidth
                                >
                                  Stop Rental (Force Logout)
                                </Button>
                                <Group gap="xs">
                                  <Button
                                    size="sm"
                                    variant="light"
                                    color="red"
                                    onClick={() => logoutAccount(account.id)}
                                    style={{ flex: 1 }}
                                  >
                                    Logout
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="light"
                                    color="orange"
                                    onClick={() => deauthorizeAccount(account.id)}
                                    style={{ flex: 1 }}
                                  >
                                    Log Off Everyone
                                  </Button>
                                </Group>
                                <Button
                                  size="sm"
                                  variant="light"
                                  color="blue"
                                  onClick={() => openChangePassword(account.id)}
                                  fullWidth
                                >
                                  Change Password
                                </Button>
                                {account.has_twofa_otp && (
                                  <Button
                                    size="sm"
                                    variant="light"
                                    onClick={() => getTwoFactorCode(account.id)}
                                    fullWidth
                                  >
                                    Get 2FA Code
                                  </Button>
                                )}
                              </Stack>
                            )}
                          </Stack>
                        </Paper>
                      )
                    })}
                  </Stack>
                </ScrollArea>
              </Paper>
            </Box>
          )}
        </Box>

        <Modal
          opened={passwordModal !== null}
          onClose={() => setPasswordModal(null)}
          title="Change Steam Password"
          centered
        >
          <Stack gap="md">
            <Text size="sm" c="dimmed">
              Enter a new password for this Steam account. The password must be at least 8 characters long.
            </Text>
            <PasswordInput
              label="New Password"
              placeholder="Enter new password"
              value={passwordModal?.newPassword || ''}
              onChange={(e) =>
                setPasswordModal((prev) => ({ ...prev, newPassword: e.target.value }))
              }
              required
            />
            <Group justify="flex-end">
              <Button variant="subtle" onClick={() => setPasswordModal(null)}>
                Cancel
              </Button>
              <Button
                onClick={changePassword}
                disabled={!passwordModal?.newPassword || passwordModal.newPassword.length < 8}
              >
                Change Password
              </Button>
            </Group>
          </Stack>
        </Modal>
      </AppShell.Main>
    </AppShell>
  )
}
