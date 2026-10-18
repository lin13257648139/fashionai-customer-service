<template>
  <div class="product-page">
    <div class="product-header">
      <div class="header-left">
        <el-button type="text" icon="el-icon-arrow-left" @click="goBack">
          ← 返回聊天
        </el-button>
        <h2>🛍️ 商品广场</h2>
      </div>

      <div class="filter-buttons">
        <el-button
          v-for="filter in filters"
          :key="filter.key"
          :type="filter.key === selectedFilter ? 'primary' : 'default'"
          @click="selectFilter(filter.key)">
          {{ filter.label }}
        </el-button>
      </div>
    </div>

    <div class="product-grid" v-if="!isLoading && products.length">
      <div class="product-card" v-for="product in filteredProducts" :key="product.name">
        <div class="product-image-wrapper" @click="showImagePreview(product)">
          <img :src="product.image_url" :alt="product.name" />
          <div class="image-overlay">
            <span class="zoom-icon">🔍</span>
          </div>
        </div>
        <div class="product-card-body">
          <div class="product-name">{{ product.name }}</div>
          <div class="product-price">{{ product.price ? `¥${product.price.replace('￥', '')}` : '价格待定' }}</div>
          <div class="product-actions">
            <el-button type="primary" plain @click="askAi(product.name)">
              🤖 询问AI助手
            </el-button>
            <el-button type="success" @click="openShop(product.shop_url)">
              🛒 购买链接
            </el-button>
          </div>
        </div>
      </div>
    </div>

    <div v-if="isLoading" class="empty-state">加载中，请稍候...</div>
    <div v-if="!isLoading && !filteredProducts.length" class="empty-state">
      当前筛选条件下暂无商品，请切换筛选条件。
    </div>

    <div v-if="errorMessage" class="error-state">{{ errorMessage }}</div>

    <!-- 图片预览弹窗 -->
    <el-dialog
      v-model="imagePreviewVisible"
      :title="previewProduct?.name || '商品图片'"
      width="80%"
      :before-close="closeImagePreview">
      <div class="image-preview-container">
        <img :src="previewProduct?.image_url" :alt="previewProduct?.name" class="preview-image" />
      </div>
      <template #footer>
        <el-button @click="closeImagePreview">关闭</el-button>
        <el-button type="primary" @click="askAi(previewProduct?.name || '')" v-if="previewProduct">
          🤖 询问AI助手
        </el-button>
        <el-button type="success" @click="openShop(previewProduct?.shop_url)" v-if="previewProduct">
          🛒 购买链接
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'

const emit = defineEmits<{
  askAi: [name: string]
  back: []
}>()

interface Product {
  name: string
  price?: string
  image_url: string
  shop_url?: string
  gender?: string
  category?: string
}

const products = ref<Product[]>([])
const selectedFilter = ref<'all' | '男装' | '女装' | '上衣' | '裤子'>('all')
const isLoading = ref(false)
const errorMessage = ref('')
const imagePreviewVisible = ref(false)
const previewProduct = ref<Product | null>(null)

type ProductFilter = typeof selectedFilter.value

const filters: Array<{ key: ProductFilter; label: string }> = [
  { key: 'all', label: '全部' },
  { key: '男装', label: '男装' },
  { key: '女装', label: '女装' },
  { key: '上衣', label: '上衣' },
  { key: '裤子', label: '裤子' }
]

const filteredProducts = computed(() => {
  if (selectedFilter.value === 'all') {
    return products.value
  }
  if (selectedFilter.value === '男装' || selectedFilter.value === '女装') {
    return products.value.filter(item => item.gender === selectedFilter.value.replace('装', ''))
  }
  return products.value.filter(item => item.category === selectedFilter.value)
})

const fetchProducts = async () => {
  isLoading.value = true
  errorMessage.value = ''
  try {
    const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000').replace(/\/$/, '')
    const response = await fetch(`${apiBaseUrl}/api/products`)
    const data = await response.json()
    products.value = Array.isArray(data.products) ? data.products : []
  } catch (error) {
    console.error('加载商品失败：', error)
    errorMessage.value = '商品列表加载失败，请稍后重试。'
  } finally {
    isLoading.value = false
  }
}

const selectFilter = (key: typeof selectedFilter.value) => {
  selectedFilter.value = key
}

const openShop = (shopUrl?: string) => {
  if (!shopUrl) return
  window.open(shopUrl, '_blank')
}

const askAi = (name: string) => {
  emit('askAi', name)
}

const goBack = () => {
  emit('back')
}

const showImagePreview = (product: Product) => {
  previewProduct.value = product
  imagePreviewVisible.value = true
}

const closeImagePreview = () => {
  imagePreviewVisible.value = false
  previewProduct.value = null
}

onMounted(() => {
  fetchProducts()
})
</script>

<style scoped>
.product-page {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  background: #ffffff;
}

.product-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px;
  border-bottom: 1px solid #e5e5e5;
  gap: 12px;
  flex-shrink: 0;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 14px;
}

.filter-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.product-grid {
  padding: 20px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px;
  overflow-y: auto;
  flex: 1;
  min-height: 0;
}

.product-card {
  display: flex;
  flex-direction: column;
  border: 1px solid #eef2f6;
  border-radius: 16px;
  overflow: hidden;
  background: #fff;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.04);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  min-height: 420px;
  overflow: hidden;
  background: #fff;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.04);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.product-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
}

.product-image-wrapper {
  position: relative;
  width: 100%;
  height: 280px;
  background: #fafbfc;
  cursor: pointer;
  overflow: hidden;
}

.product-image-wrapper:hover .image-overlay {
  opacity: 1;
}

.product-image-wrapper img {
  width: auto;
  max-width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  margin: 0 auto;
  background: white;
}

.image-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.3s ease;
}

.zoom-icon {
  font-size: 24px;
  color: white;
  background: rgba(0, 0, 0, 0.6);
  border-radius: 50%;
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.product-card-body {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
  border-top: 1px solid #f2f5f9;
  background: #fff;
}

.product-name {
  font-size: 16px;
  font-weight: 700;
  color: #333;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.product-price {
  color: #f56c6c;
  font-size: 18px;
  font-weight: 700;
}

.product-actions {
  margin-top: auto;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: space-between;
}

.product-actions .el-button {
  flex: 1;
  min-width: 0;
}

.product-price {
  color: #f56c6c;
  font-size: 18px;
  font-weight: 700;
  margin-top: 6px;
}

.empty-state,
.error-state {
  padding: 20px;
  color: #666;
  text-align: center;
}

.error-state {
  color: #f56c6c;
}

/* 图片预览弹窗样式 */
.image-preview-container {
  text-align: center;
  max-height: 70vh;
  overflow: hidden;
}

.preview-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  border-radius: 8px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .product-grid {
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    padding: 15px;
    gap: 15px;
  }

  .product-image-wrapper {
    height: 240px;
  }

  .product-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 15px;
  }

  .filter-buttons {
    width: 100%;
    justify-content: center;
  }
}

@media (max-width: 480px) {
  .product-grid {
    grid-template-columns: 1fr;
    padding: 10px;
  }

  .product-image-wrapper {
    height: 200px;
  }

  .product-card-body {
    padding: 12px;
  }

  .product-name {
    font-size: 14px;
  }

  .product-price {
    font-size: 16px;
  }
}
</style>
