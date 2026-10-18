<template>
  <div class="app-container">
    <!-- 侧边栏 -->
    <aside class="sidebar">
      <div class="logo">
        <div class="logo-icon">🤖</div>
        <h2>FashionAI</h2>
        <p>懂时尚 · 会聊天 · 能推荐</p>
      </div>

      <el-button type="primary" @click="clearChat" class="action-btn new-chat-btn">
        ✨ 新建对话
      </el-button>

      <div class="stats-card stats-card-inline stats-card-top">
        <div class="stats-item">
          <span class="stats-label">💬 对话次数</span>
          <span class="stats-value">{{ conversationCount }}</span>
        </div>
        <div class="stats-item">
          <span class="stats-label">当前情绪</span>
          <span class="stats-value">{{ emotionDisplay.emoji }} {{ emotionDisplay.text }}</span>
        </div>
      </div>

      <el-button type="primary" @click="triggerImageUpload" class="action-btn">
        📷 图片分析
      </el-button>

      <el-button type="success" @click="triggerOutfitUpload" class="action-btn">
        👗 穿搭推荐
      </el-button>

      <el-button type="warning" @click="showProductGallery" class="action-btn">
        🛍️ 商品广场
      </el-button>

      <el-button type="info" @click="openOrderQueryDialog" class="action-btn">
        📦 订单查询
      </el-button>

      <input ref="imageInput" type="file" accept="image/png,image/jpeg" style="display: none" @change="handleImageSelected" />
      <input ref="outfitInput" type="file" accept="image/png,image/jpeg" style="display: none" @change="handleOutfitSelected" />

      <el-divider />


      <div class="quick-questions">
        <h3>🚀 快捷提问</h3>
        <el-button v-for="q in quickQuestions" :key="q.label" 
                   @click="sendQuickQuestion(q.question)"
                   class="quick-btn">
          {{ q.label }}
        </el-button>
      </div>

      <div class="sidebar-footer">
        <el-button type="default" @click="openSettingsDialog" class="action-btn settings-btn">
          ⚙️ 设置
        </el-button>
      </div>
    </aside>

    <main class="chat-main">
      <div v-if="currentPage === 'chat'" class="chat-view">
        <div class="chat-header">
          <h1>FashionAI 智能客服</h1>
          <span class="online-status">● 在线</span>
        </div>

        <div class="messages-container" ref="messagesContainer">
          <div v-if="messages.length === 0" class="welcome-panel">
            <div class="welcome-logo">
              <div class="welcome-logo-icon">🤖</div>
            </div>
            <h2>你好！我是 FashionAI</h2>
            <p class="welcome-subtitle">你的智能服装助手，可以帮你：</p>

            <div class="welcome-actions">
              <button class="welcome-card" @click="sendMessageWithPayload('尺码推荐')">
                <span class="welcome-card-icon">📏</span>
                <span class="welcome-card-title">推荐合适尺码</span>
                <span class="welcome-card-desc">快速找到更合身的尺码建议</span>
              </button>

              <button class="welcome-card" @click="sendMessageWithPayload('推荐几款热销商品')">
                <span class="welcome-card-icon">🛍️</span>
                <span class="welcome-card-title">挑选心仪商品</span>
                <span class="welcome-card-desc">帮你发现更受欢迎的商品</span>
              </button>

              <button class="welcome-card" @click="sendMessageWithPayload('我要退货退款')">
                <span class="welcome-card-icon">🔄</span>
                <span class="welcome-card-title">处理退换货</span>
                <span class="welcome-card-desc">直接帮你进入售后流程</span>
              </button>

              <button class="welcome-card" @click="triggerImageUpload">
                <span class="welcome-card-icon">📷</span>
                <span class="welcome-card-title">分析服装图片</span>
                <span class="welcome-card-desc">上传图片后自动进行分析</span>
              </button>
            </div>
          </div>

          <template v-else>
            <div v-for="(msg, idx) in messages" :key="idx" 
                 :class="['message', msg.role]">
              <div class="avatar">
                {{ msg.role === 'user' ? '👤' : '🤖' }}
              </div>
              <div class="content">
                <template v-if="msg.imageUrl">
                  <div class="message-image-wrapper">
                    <img :src="msg.imageUrl" alt="上传图片" class="message-image clickable-image" @click="openImagePreview(msg.imageUrl)" />
                  </div>
                  <div v-if="msg.content" class="message-text">{{ msg.content }}</div>
                </template>
                <template v-else>
                  <template v-if="msg.role === 'assistant'">
                    <div class="rich-message">
                      <template v-for="(block, blockIdx) in parseMessageBlocks(msg.content)" :key="`${idx}-${blockIdx}`">
                        <div v-if="block.type === 'text'" class="message-text">{{ formatAssistantText(block.value) }}</div>
                        <a
                          v-else-if="block.type === 'link'"
                          :href="block.value"
                          target="_blank"
                          rel="noopener noreferrer"
                          class="message-link"
                        >
                          {{ block.label || block.value }}
                        </a>
                        <div v-else-if="block.type === 'image'" class="message-image-wrapper">
                          <img :src="resolveImageUrl(block.value)" alt="商品图片" class="message-image clickable-image" @click="openImagePreview(resolveImageUrl(block.value))" />
                        </div>
                      </template>
                    </div>
                  </template>
                  <template v-else>
                    {{ msg.content }}
                  </template>
                </template>
              </div>
            </div>
          </template>
          
          <!-- 三个点只在等待响应时显示，流式输出开始后立即消失 -->
          <div v-if="isTyping" class="message assistant">
            <div class="avatar">🤖</div>
            <div class="content typing">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>

      <div class="input-area">
        <el-input v-model="inputMessage" 
                  placeholder="输入您的问题..." 
                  @keyup.enter="sendMessage"
                  :disabled="isTyping"
                  size="large">
          <template #append>
            <el-button @click="sendMessage" :loading="isTyping" type="primary">
              发送
            </el-button>
          </template>
        </el-input>
      </div>
    </div>

    <ProductGallery
      v-else-if="currentPage === 'products'"
      @askAi="handleAskProduct"
      @back="goBackToChat"
    />

    <div v-else-if="currentPage === 'order'" class="order-page">
      <div class="order-page-header">
        <el-button text @click="goBackToChat" class="back-chat-btn">← 返回聊天</el-button>
        <h2>📦 订单详情</h2>
      </div>

      <div v-if="currentOrder" class="order-page-body">
        <div class="order-card order-card-standalone">
          <div class="order-card-header">
            <h3>📦 订单详情</h3>
            <span class="order-status-tag">{{ currentOrder.shipping?.status || '未知状态' }}</span>
          </div>

          <div class="order-meta">
            <div>订单号：{{ currentOrder.order_id }}</div>
            <div>下单时间：{{ currentOrder.order_time }}</div>
            <div>支付方式：{{ currentOrder.payment_method || '未知' }}</div>
            <div>优惠券：{{ currentOrder.coupon_used || '无' }}</div>
          </div>

          <div class="order-section-title">商品清单</div>
          <div v-for="(item, itemIdx) in currentOrder.items" :key="itemIdx" class="order-item-row">
            <div class="order-item-name">{{ item.name }}</div>
            <div class="order-item-info">x{{ item.quantity }} · ￥{{ formatAmount(item.price) }}</div>
          </div>

          <div class="order-total">总金额：￥{{ formatAmount(currentOrder.total_amount) }}</div>

          <div class="order-section-title">快递信息</div>
          <div class="order-meta">
            <div>快递公司：{{ currentOrder.shipping?.company || '未知' }}</div>
            <div>快递单号：{{ currentOrder.shipping?.tracking_number || '未知' }}</div>
            <div>物流状态：{{ currentOrder.shipping?.status || '未知' }}</div>
          </div>

          <div class="order-section-title">物流轨迹</div>
          <div v-if="currentOrder.shipping?.tracking_history?.length" class="tracking-list">
            <div v-for="(track, trackIdx) in currentOrder.shipping.tracking_history" :key="trackIdx" class="tracking-item">
              <div class="tracking-time">{{ track.time }}</div>
              <div class="tracking-status">{{ track.status }}</div>
            </div>
          </div>
          <div v-else class="tracking-empty">暂无物流轨迹</div>

          <el-button type="primary" plain class="ask-ai-btn" @click="askAiWithOrder">
            🤖 询问AI助手
          </el-button>
        </div>
      </div>
    </div>

    <el-dialog v-model="orderDialogVisible" title="📦 订单查询" width="420px">
      <el-form label-position="top">
        <el-form-item label="手机号">
          <el-input v-model="orderForm.phone" placeholder="请输入下单手机号" />
        </el-form-item>
        <el-form-item label="下单姓名">
          <el-input v-model="orderForm.name" placeholder="请输入下单姓名" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="orderDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="queryOrderLoading" @click="queryOrder">查询订单</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="settingsDialogVisible" title="⚙️ 智能客服设置" width="520px">
      <el-form label-position="top">
        <el-form-item label="速度模式">
          <el-segmented
            v-model="chatSettings.speedMode"
            :options="[
              { label: '极速', value: 'fast' },
              { label: '平衡', value: 'balanced' },
              { label: '高质量', value: 'quality' }
            ]"
          />
        </el-form-item>

        <el-form-item label="智能体模式">
          <el-segmented
            v-model="chatSettings.agentMode"
            :options="[
              { label: '自动', value: 'auto' },
              { label: 'Multi-Agent', value: 'multi' },
              { label: 'ReAct', value: 'react' }
            ]"
          />
        </el-form-item>

        <el-form-item label="回复风格">
          <el-select v-model="chatSettings.replyStyle" placeholder="请选择回复风格" style="width: 100%">
            <el-option label="简洁" value="concise" />
            <el-option label="详细" value="detailed" />
            <el-option label="客服口语化" value="friendly" />
          </el-select>
        </el-form-item>

        <el-form-item>
          <el-switch v-model="chatSettings.showReasoningSummary" />
          <span class="setting-inline-label">显示推理摘要（开发调试）</span>
        </el-form-item>
      </el-form>

      <template #footer>
        <el-button @click="resetSettings">恢复默认</el-button>
        <el-button type="primary" @click="saveSettings">保存设置</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="imagePreviewVisible" title="图片预览" width="70%" top="5vh" destroy-on-close>
      <div class="image-preview-container">
        <img v-if="previewImageUrl" :src="previewImageUrl" alt="图片预览" class="image-preview" />
      </div>
    </el-dialog>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, nextTick, onMounted, computed } from 'vue'
import { ElMessage } from 'element-plus'
import ProductGallery from './components/ProductGallery.vue'

const messages = ref<{role: string, content: string, imageUrl?: string}[]>([])
const currentPage = ref<'chat' | 'products' | 'order'>('chat')
const inputMessage = ref('')
const isTyping = ref(false)
const messagesContainer = ref<HTMLElement>()
const imageInput = ref<HTMLInputElement | null>(null)
const outfitInput = ref<HTMLInputElement | null>(null)
const orderDialogVisible = ref(false)
const queryOrderLoading = ref(false)
const settingsDialogVisible = ref(false)
const conversationCount = ref(0)
const currentEmotion = ref('中性')
const imagePreviewVisible = ref(false)
const previewImageUrl = ref('')
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000').replace(/\/$/, '')

type ChatSettings = {
  speedMode: 'fast' | 'balanced' | 'quality'
  agentMode: 'auto' | 'multi' | 'react'
  replyStyle: 'concise' | 'detailed' | 'friendly'
  showReasoningSummary: boolean
}

const defaultChatSettings: ChatSettings = {
  speedMode: 'balanced',
  agentMode: 'auto',
  replyStyle: 'friendly',
  showReasoningSummary: false
}

const chatSettings = reactive<ChatSettings>({ ...defaultChatSettings })

const orderForm = reactive({
  phone: '',
  name: ''
})

type OrderItem = {
  name: string
  price: number
  quantity: number
  color?: string
  size?: string
}

type TrackingItem = {
  time: string
  status: string
}

type OrderDetail = {
  phone: string
  name: string
  order_id: string
  order_time: string
  total_amount: number
  items: OrderItem[]
  shipping: {
    company: string
    tracking_number: string
    status: string
    tracking_history: TrackingItem[]
  }
  coupon_used?: string
  payment_method?: string
}

const currentOrder = ref<OrderDetail | null>(null)

// 会话ID - 用于保持对话上下文
const sessionId = ref(localStorage.getItem('chat_session_id') || `session_${Date.now()}`)

// 保存 session_id 到本地
localStorage.setItem('chat_session_id', sessionId.value)

const createNewSessionId = () => {
  sessionId.value = `session_${Date.now()}`
  localStorage.setItem('chat_session_id', sessionId.value)
}

const quickQuestions = [
  { label: '📏 尺码推荐', question: '尺码推荐' },
  { label: '🛠️ 售后咨询', question: '我想咨询售后问题' },
  { label: '🔄 退货退款', question: '我要退货退款' },
  { label: '💰 优惠券', question: '我可以领取什么优惠券？' }
]

const CHAT_SETTINGS_STORAGE_KEY = 'fashionai_chat_settings'

const loadSettings = () => {
  try {
    const raw = localStorage.getItem(CHAT_SETTINGS_STORAGE_KEY)
    if (!raw) return
    const parsed = JSON.parse(raw)
    chatSettings.speedMode = parsed.speedMode || defaultChatSettings.speedMode
    chatSettings.agentMode = parsed.agentMode || defaultChatSettings.agentMode
    chatSettings.replyStyle = parsed.replyStyle || defaultChatSettings.replyStyle
    chatSettings.showReasoningSummary = !!parsed.showReasoningSummary
  } catch (error) {
    console.warn('设置加载失败，使用默认配置')
  }
}

const persistSettings = () => {
  localStorage.setItem(CHAT_SETTINGS_STORAGE_KEY, JSON.stringify(chatSettings))
}

const openSettingsDialog = () => {
  settingsDialogVisible.value = true
}

const resetSettings = () => {
  chatSettings.speedMode = defaultChatSettings.speedMode
  chatSettings.agentMode = defaultChatSettings.agentMode
  chatSettings.replyStyle = defaultChatSettings.replyStyle
  chatSettings.showReasoningSummary = defaultChatSettings.showReasoningSummary
}

const saveSettings = () => {
  persistSettings()
  settingsDialogVisible.value = false
  ElMessage.success('设置已保存')
}

const formatAmount = (value: number | string | undefined) => {
  const num = Number(value ?? 0)
  return Number.isFinite(num) ? num.toFixed(2) : '0.00'
}

type MessageBlock = {
  type: 'text' | 'link' | 'image'
  label?: string
  value: string
}

const emotionMap: Record<string, { emoji: string; text: string }> = {
  愤怒: { emoji: '😠', text: '愤怒' },
  焦急: { emoji: '⏰', text: '焦急' },
  失望: { emoji: '😔', text: '失望' },
  满意: { emoji: '😊', text: '满意' },
  中性: { emoji: '😐', text: '中性' }
}

const normalizeEmotion = (emotion: string) => {
  const value = (emotion || '').trim()
  return emotionMap[value] ? value : '中性'
}

const emotionDisplay = computed(() => {
  return emotionMap[normalizeEmotion(currentEmotion.value)] ?? emotionMap['中性']!
})

const MARKDOWN_LINK_OR_IMAGE_PATTERN = /(!\[([^\]]*)\]\(([^)]+)\)|\[([^\]]+)\]\(([^)]+)\))/g
const LINK_OR_IMAGE_PATTERN = /(https?:\/\/[^\s]+|\/images\/[^\s]+)/g

const stripTrailingPunctuation = (value: string) => value.replace(/[，。！？；,!?;\]\)】}>]+$/g, '')

const isImagePathToken = (value: string) => {
  const normalized = stripTrailingPunctuation(value.trim()).toLowerCase()
  return /\.(png|jpe?g|gif|webp)$/.test(normalized)
}

const resolveImageUrl = (path: string) => {
  const normalized = stripTrailingPunctuation(path.trim())
  if (!normalized) return normalized
  if (normalized.startsWith('http://') || normalized.startsWith('https://')) return normalized
  if (normalized.startsWith('/')) return normalized
  return `/${normalized}`
}

const parseMessageBlocks = (content: string): MessageBlock[] => {
  if (!content) return []

  const blocks: MessageBlock[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null

  MARKDOWN_LINK_OR_IMAGE_PATTERN.lastIndex = 0
  while ((match = MARKDOWN_LINK_OR_IMAGE_PATTERN.exec(content)) !== null) {
    const rawToken = match[0]
    const start = match.index

    if (start > lastIndex) {
      blocks.push({ type: 'text', value: content.slice(lastIndex, start) })
    }

    const isMarkdownImage = rawToken.startsWith('![')
    const label = stripTrailingPunctuation((isMarkdownImage ? match[2] : match[4]) || '').trim()
    const url = stripTrailingPunctuation((isMarkdownImage ? match[3] : match[5]) || '').trim()

    if (url) {
      const blockType = isMarkdownImage || isImagePathToken(url) ? 'image' : 'link'
      blocks.push({
        type: blockType,
        label,
        value: url,
      })
    }

    lastIndex = start + rawToken.length
  }

  const maskedContent = content.replace(MARKDOWN_LINK_OR_IMAGE_PATTERN, token => ' '.repeat(token.length))

  LINK_OR_IMAGE_PATTERN.lastIndex = 0
  while ((match = LINK_OR_IMAGE_PATTERN.exec(maskedContent)) !== null) {
    const rawToken = match[0]
    const cleanToken = stripTrailingPunctuation(rawToken)
    const start = match.index

    if (start > lastIndex) {
      blocks.push({ type: 'text', value: content.slice(lastIndex, start) })
    }

    if (isImagePathToken(cleanToken)) {
      blocks.push({ type: 'image', value: cleanToken })
    } else {
      blocks.push({ type: 'link', value: cleanToken })
    }

    lastIndex = start + rawToken.length
  }

  if (lastIndex < content.length) {
    blocks.push({ type: 'text', value: content.slice(lastIndex) })
  }

  return blocks.filter(block => block.value !== '')
}

const formatAssistantText = (text: string) => {
  const value = (text || '').trim()
  if (!value) return value
  if (value.includes('\n')) return value
  if (value.length < 80) return value

  return value
    .replace(/([。！？!?；;])\s*(?=(第一款|第二款|第三款|最后|如果您|需要了解更多|需要更多|需要|再告诉我|另外|此外))/g, '$1\n\n')
    .replace(/([。！？!?；;])\s*/g, '$1\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

const sendMessageWithPayload = async (
  requestMessage: string,
  displayMessage?: string,
  options?: { bypassOrderQueryDialog?: boolean }
) => {
  if (!requestMessage.trim() || isTyping.value) return

  const userMsg = requestMessage
  conversationCount.value += 1
  messages.value.push({ role: 'user', content: displayMessage || userMsg })
  if (!displayMessage) {
    inputMessage.value = ''
  }
  
  await nextTick()
  scrollToBottom()
  
  // 显示三个点（等待后端响应）
  isTyping.value = true
  
  try {
    // 使用流式接口
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: userMsg,
        session_id: sessionId.value,
        bypass_order_query_dialog: options?.bypassOrderQueryDialog === true,
        chat_settings: {
          speed_mode: chatSettings.speedMode,
          agent_mode: chatSettings.agentMode,
          reply_style: chatSettings.replyStyle,
          show_reasoning_summary: chatSettings.showReasoningSummary
        }
      })
    })
    
    const reader = response.body?.getReader()
    const decoder = new TextDecoder()
    let assistantMsg = ''
    let buffer = ''
    
    // 收到第一个数据块时，立即隐藏三个点
    let hasStarted = false
    
    // 添加一个空的助手消息占位
    messages.value.push({ role: 'assistant', content: '' })
    const lastIndex = messages.value.length - 1
    
    if (!reader) {
      throw new Error('无法读取响应流')
    }
    
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      
      for (const rawLine of lines) {
        const line = rawLine.trim()
        if (!line.startsWith('data: ')) continue
        try {
          const data = JSON.parse(line.slice(6).trim())
          if (data.chunk) {
            if (!hasStarted) {
              hasStarted = true
              isTyping.value = false
            }
            assistantMsg += data.chunk
            const currentMessage = messages.value[lastIndex]
            if (currentMessage) currentMessage.content = assistantMsg
            await nextTick()
            scrollToBottom()
          }
          if (data.image_url) {
            messages.value.push({
              role: 'assistant',
              content: '👆 这是为您生成的搭配效果图',
              imageUrl: String(data.image_url)
            })
            await nextTick()
            scrollToBottom()
          }
          if (data.emotion) {
            currentEmotion.value = normalizeEmotion(String(data.emotion))
          }
          if (data.error) {
            throw new Error(data.error)
          }
          if (data.done) {
            isTyping.value = false
          }
        } catch (err) {
          // 忽略解析错误，等待下一个完整事件
        }
      }
    }
    
    // 如果缓冲区里还有最后一行，尝试解析它
    if (buffer.trim().startsWith('data: ')) {
      try {
        const data = JSON.parse(buffer.slice(6).trim())
        if (data.chunk) {
          assistantMsg += data.chunk
          const currentMessage = messages.value[lastIndex]
          if (currentMessage) currentMessage.content = assistantMsg
        }
        if (data.emotion) {
          currentEmotion.value = normalizeEmotion(String(data.emotion))
        }
        if (data.error) {
          throw new Error(data.error)
        }
      } catch (err) {
        // 不影响最终展示
      }
    }
    
    // 确保最终三个点隐藏
    isTyping.value = false
    
  } catch (error) {
    console.error('请求失败:', error)
    ElMessage.error('请求失败，请确保后端已启动')
    // 移除空的助手消息
    messages.value.pop()
    isTyping.value = false  // 出错时也要隐藏三个点
  } finally {
    isTyping.value = false
    await nextTick()
    scrollToBottom()
  }
}

// 发送消息（支持流式输出）
const sendMessage = async () => {
  if (!inputMessage.value.trim()) return
  await sendMessageWithPayload(inputMessage.value)
}

const openOrderQueryDialog = () => {
  orderDialogVisible.value = true
}

const queryOrder = async () => {
  if (!orderForm.phone.trim() || !orderForm.name.trim()) {
    ElMessage.warning('请输入手机号和下单姓名')
    return
  }

  queryOrderLoading.value = true
  try {
    const response = await fetch(`${API_BASE_URL}/query_order`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        phone: orderForm.phone.trim(),
        name: orderForm.name.trim(),
        session_id: sessionId.value
      })
    })

    const data = await response.json()
    if (data.success && data.order) {
      currentOrder.value = data.order
      orderDialogVisible.value = false
      currentPage.value = 'order'
      ElMessage.success('订单查询成功')
    } else {
      ElMessage.warning(data.message || '未找到订单，请检查手机号和姓名')
    }
  } catch (error) {
    ElMessage.error('订单查询失败，请稍后重试')
  } finally {
    queryOrderLoading.value = false
  }
}

const askAiWithOrder = async () => {
  if (!currentOrder.value) return

  const trackingNumber = currentOrder.value.shipping?.tracking_number || '未知'
  const prompt = `我的订单【快递单号：${trackingNumber}】`
  currentPage.value = 'chat'
  await nextTick()
  await sendMessageWithPayload(
    prompt,
    prompt,
    { bypassOrderQueryDialog: true }
  )
}

// 快捷提问
const triggerImageUpload = () => {
  imageInput.value?.click()
}

const handleImageSelected = async (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return
  target.value = ''
  await uploadImage(file)
}

const triggerOutfitUpload = () => {
  outfitInput.value?.click()
}

const showProductGallery = () => {
  currentPage.value = 'products'
}

const goBackToChat = () => {
  currentPage.value = 'chat'
}

const handleAskProduct = async (name: string) => {
  inputMessage.value = `我想了解${name}`
  currentPage.value = 'chat'
  await nextTick()
  sendMessage()
}

const handleOutfitSelected = async (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return
  target.value = ''
  await recommendOutfit(file)
}

const uploadImage = async (file: File) => {
  if (isTyping.value) return
  const validTypes = ['image/jpeg', 'image/png']
  if (!validTypes.includes(file.type)) {
    ElMessage.warning('仅支持 JPG/PNG 图片')
    return
  }

  isTyping.value = true
  const imageUrl = URL.createObjectURL(file)
  messages.value.push({
    role: 'user',
    content: '已上传一张服装图片，请查看。',
    imageUrl
  })
  await nextTick()
  scrollToBottom()

  try {
    const formData = new FormData()
    formData.append('image', file)
    formData.append('session_id', sessionId.value)

    const response = await fetch(`${API_BASE_URL}/chat/image/stream`, {
      method: 'POST',
      body: formData
    })

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('无法读取响应流')
    }

    messages.value.push({ role: 'assistant', content: '' })
    const assistantIndex = messages.value.length - 1
    let assistantText = ''

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const rawLine of lines) {
        const line = rawLine.trim()
        if (!line.startsWith('data: ')) continue
        try {
          const payload = JSON.parse(line.slice(6).trim())
          if (payload.error) {
            throw new Error(payload.error)
          }
          if (payload.chunk) {
            assistantText += payload.chunk
            const currentMessage = messages.value[assistantIndex]
            if (currentMessage) currentMessage.content = assistantText
            await nextTick()
            scrollToBottom()
          }
          if (payload.done) {
            isTyping.value = false
          }
        } catch (err) {
          throw err
        }
      }
    }

    await nextTick()
    scrollToBottom()
  } catch (error) {
    console.error('图片上传失败:', error)
    ElMessage.error('图片上传失败，请重试')
    messages.value.pop()
  } finally {
    isTyping.value = false
  }
}

const recommendOutfit = async (file: File) => {
  if (isTyping.value) return
  const validTypes = ['image/jpeg', 'image/png']
  if (!validTypes.includes(file.type)) {
    ElMessage.warning('仅支持 JPG/PNG 图片')
    return
  }

  isTyping.value = true
  const imageUrl = URL.createObjectURL(file)
  messages.value.push({
    role: 'user',
    content: '已上传一张服装图片，请求穿搭推荐。',
    imageUrl
  })
  await nextTick()
  scrollToBottom()

  const assistantIndex = messages.value.push({ role: 'assistant', content: '正在分析您的穿搭，请稍候...' }) - 1

  const readFileAsBase64 = (file: File) => {
    return new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        const result = reader.result as string
        const base64 = result.split(',')[1]
        if (!base64) {
          reject(new Error('图片编码失败'))
          return
        }
        resolve(base64)
      }
      reader.onerror = () => reject(new Error('图片读取失败'))
      reader.readAsDataURL(file)
    })
  }

  try {
    const base64 = await readFileAsBase64(file)
    const response = await fetch(`${API_BASE_URL}/chat/outfit_recommend/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_base64: base64,
        mime_type: file.type,
        session_id: sessionId.value
      })
    })

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('无法读取响应流')
    }

    let assistantText = ''
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const rawLine of lines) {
        const line = rawLine.trim()
        if (!line.startsWith('data: ')) continue
        try {
          const payload = JSON.parse(line.slice(6).trim())
          if (payload.error) {
            throw new Error(payload.error)
          }
          if (payload.chunk) {
            assistantText += payload.chunk
            const currentMessage = messages.value[assistantIndex]
            if (currentMessage) currentMessage.content = assistantText
            await nextTick()
            scrollToBottom()
          }
          if (payload.image_url) {
            messages.value.push({
              role: 'assistant',
              content: '👆 这是为您生成的搭配效果图',
              imageUrl: payload.image_url
            })
            await nextTick()
            scrollToBottom()
          }
          if (payload.done) {
            isTyping.value = false
          }
        } catch (err) {
          throw err
        }
      }
    }

    await nextTick()
    scrollToBottom()
  } catch (error) {
    console.error('穿搭推荐失败:', error)
    const message = error instanceof Error ? error.message : '穿搭推荐失败，请重试'
    ElMessage.error(message)
    const currentMessage = messages.value[assistantIndex]
    if (currentMessage) currentMessage.content = message
  } finally {
    isTyping.value = false
  }
}

const sendQuickQuestion = (question: string) => {
  inputMessage.value = question
  sendMessage()
}

const openImagePreview = (url: string) => {
  if (!url) return
  previewImageUrl.value = url
  imagePreviewVisible.value = true
}

// 清空对话
const clearChat = async () => {
  const oldSessionId = sessionId.value
  createNewSessionId()

  try {
    await fetch(`${API_BASE_URL}/chat/clear`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: oldSessionId })
    })
  } catch (error) {
    // 即使后端清空失败，新会话也不会继承旧上下文
    console.warn('旧会话清理失败，已切换到新会话', error)
  }

  messages.value = []
  conversationCount.value = 0
  currentEmotion.value = '中性'
  currentOrder.value = null
  orderForm.phone = ''
  orderForm.name = ''
  inputMessage.value = ''
  currentPage.value = 'chat'
  ElMessage.success('已开始新对话')
}

// 滚动到底部
const scrollToBottom = () => {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

// 组件挂载时，如果有历史消息可以加载（可选）
onMounted(() => {
  loadSettings()
  console.log('会话ID:', sessionId.value)
})
</script>

<style scoped>
* { margin: 0; padding: 0; box-sizing: border-box; }

.app-container {
  display: flex;
  height: 100vh;
  width: 100%;
  overflow: hidden;  /* 关键：防止双滚动条 */
  background: #f5f5f5;
}

.sidebar {
  width: 280px;
  display: flex;
  flex-direction: column;
  background: linear-gradient(180deg, #f8fcff 0%, #eff6ff 45%, #e8f2ff 100%);
  border-right: 1px solid #e5e5e5;
  padding: 20px;
  overflow-y: auto;  /* 侧边栏独立滚动 */
  height: 100%;
  flex-shrink: 0;
}

.clickable-image {
  cursor: zoom-in;
}

.image-preview-container {
  display: flex;
  justify-content: center;
}

.image-preview {
  max-width: 100%;
  max-height: 75vh;
  object-fit: contain;
}

/* 侧边栏滚动条美化（可选） */
.sidebar::-webkit-scrollbar {
  width: 4px;
}
.sidebar::-webkit-scrollbar-track {
  background: #f1f1f1;
}
.sidebar::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 4px;
}
.sidebar::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

.logo {
  text-align: center;
  margin-bottom: 18px;
}

.logo-icon {
  width: 64px;
  height: 64px;
  background: radial-gradient(circle at 30% 30%, #ffffff 0%, #dbeafe 45%, #bfdbfe 100%);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 31px;
  margin: 0 auto 10px;
  box-shadow: 0 8px 18px rgba(59, 130, 246, 0.18);
}

.logo h2 {
  margin: 0;
  color: #1f2937;
  letter-spacing: 0.4px;
}

.logo p {
  margin: 4px 0 0;
  color: #64748b;
  font-size: 12px;
}

.action-btn {
  width: 92%;
  margin-bottom: 12px;
  margin-left: auto;
  margin-right: auto;
  text-align: center;
  border-radius: 12px;
  box-shadow: 0 4px 10px rgba(15, 23, 42, 0.08);
}

.new-chat-btn {
  margin-bottom: 10px;
}

.stats-card-top {
  margin-bottom: 18px;
}

.message-image-wrapper {
  margin-bottom: 8px;
}

.message-image {
  max-width: 220px;
  max-height: 220px;
  border-radius: 14px;
  display: block;
  object-fit: cover;
  border: 1px solid rgba(0,0,0,0.08);
}

.message-text {
  white-space: pre-wrap;
  word-break: break-word;
}

.order-card {
  margin: 16px 20px 0;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 14px;
  background: #fafafa;
}

.order-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.order-card-header h3 {
  margin: 0;
  font-size: 16px;
  color: #111827;
}

.order-status-tag {
  font-size: 12px;
  color: #1d4ed8;
  background: #dbeafe;
  border-radius: 999px;
  padding: 4px 10px;
}

.order-meta {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px 12px;
  font-size: 13px;
  color: #374151;
  margin-bottom: 10px;
}

.order-section-title {
  font-size: 13px;
  font-weight: 600;
  color: #1f2937;
  margin: 10px 0 6px;
}

.order-item-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
  padding: 6px 0;
  border-bottom: 1px dashed #e5e7eb;
}

.order-item-name {
  color: #111827;
}

.order-item-info {
  color: #6b7280;
}

.order-total {
  text-align: right;
  font-weight: 700;
  color: #111827;
  margin-top: 8px;
}

.tracking-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.tracking-item {
  display: grid;
  grid-template-columns: 160px 1fr;
  gap: 10px;
  font-size: 12px;
}

.tracking-time {
  color: #6b7280;
}

.tracking-status {
  color: #111827;
}

.tracking-empty {
  font-size: 12px;
  color: #6b7280;
}

.ask-ai-btn {
  margin-top: 12px;
}

.rich-message {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.message-link {
  color: #2563eb;
  text-decoration: underline;
  word-break: break-all;
}

.message-link:hover {
  color: #1d4ed8;
}

.user-info h3, .quick-questions h3 {
  font-size: 14px;
  margin-bottom: 10px;
  color: #666;
}

.quick-btn {
  display: block;
  width: 92%;
  margin-bottom: 10px;
  margin-left: auto;
  margin-right: auto;
  text-align: center;
  border-radius: 10px;
}

.quick-questions {
  width: 100%;
  margin-top: 16px;
}

.quick-questions h3 {
  text-align: center;
}

.sidebar-footer {
  margin-top: auto;
  padding-top: 12px;
  border-top: 1px dashed #e5e7eb;
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.settings-btn {
  width: 92%;
  margin-bottom: 0;
  box-shadow: none;
}

.stats-card {
  width: 92%;
  margin: 0 auto;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(148, 163, 184, 0.35);
  border-radius: 10px;
  padding: 8px 10px;
}

.stats-card-inline {
  margin-top: 6px;
}

.stats-card-top {
  margin-bottom: 14px;
}

.stats-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  color: #334155;
  line-height: 1.6;
}

.stats-item + .stats-item {
  margin-top: 4px;
  padding-top: 4px;
  border-top: 1px dashed rgba(148, 163, 184, 0.35);
}

.stats-label {
  color: #64748b;
}

.stats-value {
  font-weight: 600;
  color: #1e293b;
}

.setting-inline-label {
  margin-left: 10px;
  color: #374151;
  font-size: 14px;
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: white;
  min-width: 0;  /* 关键：防止flex子项溢出 */
  height: 100%;
  overflow: hidden;  /* 关键：让内部messages-container独立滚动 */
}

.chat-view {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0; /* 关键：让子元素正确撑满高度 */
}

.order-page {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  background: white;
}

.order-page-header {
  padding: 16px 20px;
  border-bottom: 1px solid #e5e5e5;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}

.order-page-header h2 {
  margin: 0;
  font-size: 18px;
  color: #111827;
}

.back-chat-btn {
  font-size: 14px;
}

.order-page-body {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.order-card-standalone {
  margin: 0;
  max-width: 960px;
}

.chat-header {
  padding: 20px;
  border-bottom: 1px solid #e5e5e5;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-shrink: 0;  /* 固定头部，不参与滚动 */
}

.chat-header h1 { font-size: 20px; color: #333; }
.online-status { color: #10a37f; font-size: 12px; }

.messages-container {
  flex: 1;
  overflow-y: auto;  /* 只有这里滚动，解决双滚动条问题 */
  padding: 20px;
  min-height: 0;  /* 关键：flex子项溢出限制 */
}

.welcome-panel {
  min-height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 24px 12px 32px;
}

.welcome-logo {
  width: 86px;
  height: 86px;
  border-radius: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 18px;
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.16), rgba(16, 185, 129, 0.14));
  border: 1px solid rgba(148, 163, 184, 0.22);
  box-shadow: 0 14px 30px rgba(15, 23, 42, 0.08);
}

.welcome-logo-icon {
  font-size: 40px;
  line-height: 1;
}

.welcome-panel h2 {
  font-size: 28px;
  color: #0f172a;
  margin-bottom: 8px;
  font-weight: 700;
}

.welcome-subtitle {
  color: #64748b;
  font-size: 15px;
  margin-bottom: 22px;
}

.welcome-actions {
  width: min(100%, 680px);
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.welcome-card {
  border: 1px solid rgba(148, 163, 184, 0.22);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(248, 250, 252, 0.92));
  border-radius: 18px;
  padding: 18px 16px;
  text-align: left;
  cursor: pointer;
  box-shadow: 0 10px 22px rgba(15, 23, 42, 0.06);
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 118px;
}

.welcome-card:hover {
  transform: translateY(-2px);
  border-color: rgba(59, 130, 246, 0.28);
  box-shadow: 0 16px 28px rgba(15, 23, 42, 0.1);
}

.welcome-card-icon {
  font-size: 24px;
}

.welcome-card-title {
  font-size: 16px;
  font-weight: 700;
  color: #0f172a;
}

.welcome-card-desc {
  font-size: 13px;
  color: #64748b;
  line-height: 1.5;
}

@media (max-width: 720px) {
  .welcome-actions {
    grid-template-columns: 1fr;
  }

  .welcome-panel h2 {
    font-size: 24px;
  }
}

/* 消息区域滚动条美化 */
.messages-container::-webkit-scrollbar {
  width: 6px;
}
.messages-container::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}
.messages-container::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}
.messages-container::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

.message {
  display: flex;
  margin-bottom: 20px;
  animation: fadeIn 0.3s ease;
}

.message.user { flex-direction: row-reverse; }
.message.user .content {
  background: #667eea;
  color: white;
  border-radius: 18px 18px 4px 18px;
}
.message.assistant .content {
  background: #f0f0f0;
  color: #333;
  border-radius: 18px 18px 18px 4px;
}

.avatar {
  width: 36px;
  height: 36px;
  background: #e5e5e5;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 10px;
  flex-shrink: 0;
}

.content {
  max-width: 70%;
  padding: 12px 16px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}

/* 三个点打字动画 */
.typing span {
  display: inline-block;
  width: 8px;
  height: 8px;
  background: #999;
  border-radius: 50%;
  margin: 0 2px;
  animation: bounce 1.4s infinite ease-in-out;
}
.typing span:nth-child(1) { animation-delay: 0s; }
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
  0%, 60%, 100% {
    transform: translateY(0);
    opacity: 0.6;
  }
  30% {
    transform: translateY(-8px);
    opacity: 1;
  }
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.input-area {
  padding: 20px;
  border-top: 1px solid #e5e5e5;
  background: white;
  flex-shrink: 0;  /* 固定输入区域，不参与滚动 */
}
</style>
