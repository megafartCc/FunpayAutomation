import { useEffect, useMemo, useState } from 'react'
import {
  AppShell,
  Avatar,
  Badge,
  Box,
  Button,
  Group,
  Paper,
  ScrollArea,
  Stack,
  Text,
  Textarea,
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
  const [error, setError] = useState('')

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
    const timer = setInterval(loadDialogs, 30000)
    return () => clearInterval(timer)
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
      const data = await api('/api/dialogs')
      setDialogs(data)
    } catch (e) {
      setError(e.message)
    }
  }

  const syncMessages = async (nodeId, chatNode) => {
    if (!nodeId && !chatNode) return null
    try {
      const data = await api('/api/messages/sync', {
        method: 'POST',
        body: JSON.stringify({ node: nodeId || null, chat_node: chatNode || null }),
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

    const sync = await syncMessages(dialog.user_id || null, dialog.node_id)
    const resolvedUserId = sync?.user_id || dialog.user_id
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
      const msgs = await api(`/api/messages?node=${encodeURIComponent(nodeId)}&limit=200`)
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
      await syncMessages(activeNode, activeChatNode)
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
        <Box
          style={{
            height: '100%',
            display: 'grid',
            gridTemplateColumns: isNarrow ? '1fr' : '320px minmax(520px, 1fr)',
            gridTemplateRows: isNarrow ? 'minmax(0, 1fr) minmax(0, 1fr)' : '1fr',
            gap: 16,
            maxWidth: 1200,
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
                        <Avatar src={d.avatar} radius="xl" color="gray">
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
                minRows={2}
                maxRows={4}
                style={{ flex: 1 }}
              />
              <Button onClick={sendMessage} disabled={!activeNode} style={{ alignSelf: 'stretch' }}>
                Send
              </Button>
            </Group>
          </Paper>
        </Box>
      </AppShell.Main>
    </AppShell>
  )
}
