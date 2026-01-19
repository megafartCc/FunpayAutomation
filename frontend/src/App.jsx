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
        return // Success
      } catch (e) {
        console.error(`Failed to load dialogs (attempt ${i + 1}/${retries}):`, e)
        if (i === retries - 1) {
          // Last attempt failed - don't block, just log
          console.warn('Dialogs failed to load after retries')
        } else {
          await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)))
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
        return // Success
      } catch (e) {
        console.error(`Failed to load lots (attempt ${i + 1}/${retries}):`, e)
        if (i === retries - 1) {
          setError('Failed to load lots: ' + e.message)
          setLoadingLots(false)
        } else {
          await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)))
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
    </AppShell>
  )
}
