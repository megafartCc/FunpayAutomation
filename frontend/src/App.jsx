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
  Loader,
  Modal,
  Paper,
  PasswordInput,
  ScrollArea,
  Stack,
  Text,
  Textarea,
  TextInput,
  Tooltip,
} from '@mantine/core'
import { useMediaQuery } from '@mantine/hooks'

const api = async (path, options = {}) => {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  
  // Read response body once
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
  
  // Parse the already-read text
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
  const [loggingIn, setLoggingIn] = useState({}) // Track which account is logging in
  const [changePasswordModal, setChangePasswordModal] = useState(null) // { accountId, newPassword }

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
    setLoggingIn((prev) => ({ ...prev, [accountId]: true }))
    setError('')
    try {
      const code = accountGuards[accountId] || ''
      await api(`/api/accounts/${accountId}/login`, {
        method: 'POST',
        body: JSON.stringify({ 
          guard_code: code,
          email_code: code, // Also send as email_code in case it's needed
        }),
      })
      // Clear the code field on success
      setAccountGuards((prev) => ({ ...prev, [accountId]: '' }))
      await loadAccounts()
    } catch (e) {
      const errorMsg = e.message || 'Login failed'
      setError(errorMsg)
      // Reload accounts to get updated status
      await loadAccounts()
    } finally {
      setLoggingIn((prev) => ({ ...prev, [accountId]: false }))
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
                    {accounts.map((acc) => {
                      const isLoggingIn = loggingIn[acc.id] || false
                      const isOnline = acc.login_status === 'online'
                      const needsEmailCode = acc.login_status === 'guard:email'
                      const needs2FA = acc.login_status === 'guard:twofactor'
                      const hasError = acc.login_status?.startsWith('error:')
                      const isIdle = !acc.login_status || acc.login_status === 'idle'
                      
                      const getStatusColor = () => {
                        if (isOnline) return 'green'
                        if (needsEmailCode || needs2FA) return 'yellow'
                        if (hasError) return 'red'
                        return 'gray'
                      }
                      
                      const getStatusText = () => {
                        if (isOnline) return 'Online'
                        if (needsEmailCode) return 'Email Code Required'
                        if (needs2FA) return '2FA Code Required'
                        if (hasError) {
                          const errorType = acc.login_status?.split(':')[1] || 'unknown'
                          const errorMap = {
                            'invalid_password': 'Invalid Password',
                            'invalid_auth_code': 'Invalid Code',
                            'account_not_found': 'Account Not Found',
                            'rate_limit': 'Rate Limited',
                            'init': 'Init Failed',
                            'connect': 'Connection Failed',
                            'exception': 'Login Error',
                          }
                          return errorMap[errorType] || 'Error'
                        }
                        return 'Idle'
                      }
                      
                      return (
                        <Paper 
                          key={acc.id} 
                          withBorder 
                          radius="md" 
                          p="md"
                          style={{
                            borderColor: isOnline ? 'var(--mantine-color-green-6)' : 
                                        needsEmailCode || needs2FA ? 'var(--mantine-color-yellow-6)' :
                                        hasError ? 'var(--mantine-color-red-6)' : undefined
                          }}
                        >
                          <Stack gap="sm">
                            {/* Header with account info and status */}
                            <Group justify="space-between" wrap="nowrap">
                              <Box style={{ flex: 1, minWidth: 0 }}>
                                <Group gap="xs" mb={4}>
                                  <Text size="sm" fw={600} truncate>
                                    {acc.label || acc.username}
                                  </Text>
                                  {isLoggingIn && <Loader size="xs" />}
                                </Group>
                                <Text size="xs" c="dimmed" truncate>
                                  {acc.steam_id ? `SteamID: ${acc.steam_id}` : 'SteamID: Not set'}
                                </Text>
                              </Box>
                              <Badge color={getStatusColor()} variant="light" size="sm">
                                {getStatusText()}
                              </Badge>
                            </Group>
                            
                            {/* Status-specific messages */}
                            {needsEmailCode && (
                              <Alert color="yellow" variant="light" p="xs" radius="sm">
                                <Text size="xs">
                                  📧 <strong>Email Code Required</strong><br />
                                  Check your email inbox for a Steam Guard code. Enter it below and click Login.
                                </Text>
                              </Alert>
                            )}
                            
                            {needs2FA && (
                              <Alert color="yellow" variant="light" p="xs" radius="sm">
                                <Text size="xs">
                                  🔐 <strong>2FA Code Required</strong><br />
                                  Enter your Steam Guard mobile authenticator code below.
                                </Text>
                              </Alert>
                            )}
                            
                            {hasError && (
                              <Alert color="red" variant="light" p="xs" radius="sm">
                                <Text size="xs">
                                  ❌ <strong>Login Failed</strong><br />
                                  {acc.login_status === 'error:invalid_password' && 'Invalid username or password. Please check your credentials.'}
                                  {acc.login_status === 'error:invalid_auth_code' && 'The code you entered is incorrect or expired. Please try again with a fresh code.'}
                                  {acc.login_status === 'error:rate_limit' && 'Too many login attempts. Please wait a few minutes before trying again.'}
                                  {acc.login_status === 'error:account_not_found' && 'Steam account not found. Check your username.'}
                                  {!acc.login_status?.includes('invalid_password') && 
                                   !acc.login_status?.includes('invalid_auth_code') && 
                                   !acc.login_status?.includes('rate_limit') && 
                                   !acc.login_status?.includes('account_not_found') && 
                                   'An error occurred during login. Please try again.'}
                                </Text>
                              </Alert>
                            )}
                            
                            {isOnline && (
                              <Alert color="green" variant="light" p="xs" radius="sm">
                                <Text size="xs">
                                  ✅ <strong>Successfully Logged In</strong><br />
                                  Account is online and ready to use.
                                </Text>
                              </Alert>
                            )}
                            
                            {/* Code input and action buttons */}
                            {!isOnline && (
                              <Stack gap="xs">
                                <TextInput
                                  size="sm"
                                  placeholder={
                                    needsEmailCode
                                      ? 'Enter 5-character email code (e.g., A1B2C)'
                                      : needs2FA
                                      ? 'Enter 5-character 2FA code'
                                      : 'Enter code if required (leave empty for first attempt)'
                                  }
                                  value={accountGuards[acc.id] || ''}
                                  onChange={(e) => {
                                    const value = e.target.value.toUpperCase().slice(0, 5)
                                    setAccountGuards((prev) => ({ ...prev, [acc.id]: value }))
                                  }}
                                  disabled={isLoggingIn}
                                  maxLength={5}
                                  style={{ fontFamily: 'monospace', letterSpacing: '2px' }}
                                />
                                <Group gap="xs">
                                  <Button
                                    size="sm"
                                    onClick={() => loginAccount(acc.id)}
                                    disabled={isLoggingIn || isOnline}
                                    loading={isLoggingIn}
                                    style={{ flex: 1 }}
                                  >
                                    {isLoggingIn
                                      ? 'Logging in...'
                                      : needsEmailCode || needs2FA
                                      ? 'Submit Code'
                                      : 'Login'}
                                  </Button>
                                </Group>
                              </Stack>
                            )}
                            
                            {/* Control Panel when online */}
                            {isOnline && (
                              <Stack gap="xs" mt="sm">
                                <Divider label="Control Panel" labelPosition="center" />
                                <Group gap="xs">
                                  <Button
                                    size="sm"
                                    variant="light"
                                    color="red"
                                    onClick={() => logoutAccount(acc.id)}
                                    style={{ flex: 1 }}
                                  >
                                    Logout
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="light"
                                    color="orange"
                                    onClick={async () => {
                                      try {
                                        await api(`/api/accounts/${acc.id}/deauthorize`, {
                                          method: 'POST',
                                        })
                                        setError('')
                                        await loadAccounts()
                                      } catch (e) {
                                        setError(e.message)
                                      }
                                    }}
                                    style={{ flex: 1 }}
                                  >
                                    Log Off Everyone
                                  </Button>
                                </Group>
                                <Button
                                  size="sm"
                                  variant="light"
                                  color="blue"
                                  onClick={() => setChangePasswordModal({ accountId: acc.id, newPassword: '' })}
                                  fullWidth
                                >
                                  Change Password
                                </Button>
                                {acc.has_twofa_otp && (
                                  <Button
                                    size="sm"
                                    variant="light"
                                    onClick={async () => {
                                      try {
                                        const data = await api(`/api/accounts/${acc.id}/code`)
                                        if (data.code) {
                                          setAccountGuards((prev) => ({ ...prev, [acc.id]: data.code }))
                                          setError('')
                                        }
                                      } catch (e) {
                                        setError(e.message)
                                      }
                                    }}
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
      </AppShell.Main>
      
      {/* Change Password Modal */}
      <Modal
        opened={changePasswordModal !== null}
        onClose={() => setChangePasswordModal(null)}
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
            value={changePasswordModal?.newPassword || ''}
            onChange={(e) =>
              setChangePasswordModal((prev) => ({ ...prev, newPassword: e.target.value }))
            }
            required
          />
          <Group justify="flex-end">
            <Button variant="subtle" onClick={() => setChangePasswordModal(null)}>
              Cancel
            </Button>
            <Button
              onClick={async () => {
                if (!changePasswordModal?.accountId || !changePasswordModal?.newPassword) return
                if (changePasswordModal.newPassword.length < 8) {
                  setError('Password must be at least 8 characters long.')
                  return
                }
                try {
                  await api(`/api/accounts/${changePasswordModal.accountId}/change-password`, {
                    method: 'POST',
                    body: JSON.stringify({ new_password: changePasswordModal.newPassword }),
                  })
                  setError('')
                  setChangePasswordModal(null)
                  await loadAccounts()
                } catch (e) {
                  setError(e.message)
                }
              }}
              disabled={!changePasswordModal?.newPassword || changePasswordModal.newPassword.length < 8}
            >
              Change Password
            </Button>
          </Group>
        </Stack>
      </Modal>
    </AppShell>
  )
}
