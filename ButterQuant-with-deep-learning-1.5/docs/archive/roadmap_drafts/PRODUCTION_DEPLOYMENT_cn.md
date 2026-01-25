# ButterQuant 生产环境部署方案

## 📊 问题诊断

### 当前架构的性能瓶颈

您的顾问说得对。当前架构存在以下问题：

1. **Flask 单线程阻塞**
   - Flask 默认是单进程单线程，即使用 `gunicorn` 也只能处理有限并发
   - 您的 `analyzer.py` 包含大量计算密集型操作（ARIMA、GARCH、FFT、Black-Scholes）
   - 单次分析耗时 **5-15秒**，高并发下会导致请求排队

2. **实时计算的性能问题**
   ```python
   # app.py 第135行 - 每次请求都触发完整分析
   @app.route('/api/analyze', methods=['POST'])
   def analyze():
       analyzer = ButterflyAnalyzer(ticker)  # 实时下载数据
       result = analyzer.full_analysis()     # 5-15秒计算
   ```

3. **数据库并发瓶颈**
   - SQLite 在高并发写入时会锁表
   - 您的 `daily_scanner.py` 使用多线程写入，但 SQLite 不适合生产环境

4. **无缓存机制**
   - 相同股票的重复请求会重复计算
   - 没有 CDN 或边缘缓存

---

## 🎯 生产级解决方案（3个方案）

### 方案 1：最小改动方案（推荐用于快速上线）

**核心思路**：保持 Python 后端，但将计算异步化 + 增加缓存

#### 架构图
```
用户请求 → Nginx → Flask (API Gateway)
                      ↓
                   Redis 缓存 (检查是否有缓存结果)
                      ↓ (cache miss)
                   Celery 任务队列
                      ↓
                   Celery Workers (多进程计算)
                      ↓
                   PostgreSQL (存储结果)
```

#### 技术栈调整
| 组件 | 当前 | 升级后 |
|------|------|--------|
| Web服务器 | Flask Dev Server | **Nginx + Gunicorn (4-8 workers)** |
| 数据库 | SQLite | **PostgreSQL** (支持高并发) |
| 缓存 | 无 | **Redis** (缓存分析结果) |
| 任务队列 | 无 | **Celery + Redis** (异步计算) |
| 部署 | 本地 | **Docker Compose** |

#### 改动清单

**1. 将计算密集型任务移到 Celery**
```python
# backend/tasks.py (新建)
from celery import Celery
import redis
import json

celery_app = Celery('butterquant', broker='redis://localhost:6379/0')
redis_client = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)

@celery_app.task
def analyze_ticker_async(ticker):
    """异步分析任务"""
    from analyzer import ButterflyAnalyzer
    
    # 检查缓存 (5分钟有效期)
    cache_key = f"analysis:{ticker}"
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 执行分析
    analyzer = ButterflyAnalyzer(ticker)
    result = analyzer.full_analysis()
    
    # 存入缓存
    redis_client.setex(cache_key, 300, json.dumps(result))
    return result
```

**2. API 改为异步模式**
```python
# backend/app.py 修改
from tasks import analyze_ticker_async
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    ticker = request.json.get('ticker', 'AAPL').upper()
    
    # 先检查 Redis 缓存
    cache_key = f"analysis:{ticker}"
    cached = redis_client.get(cache_key)
    if cached:
        return jsonify({'success': True, 'data': json.loads(cached), 'from_cache': True})
    
    # 提交异步任务
    task = analyze_ticker_async.delay(ticker)
    
    # 返回任务ID，前端轮询
    return jsonify({'success': True, 'task_id': task.id, 'status': 'processing'})

@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """查询任务状态"""
    task = analyze_ticker_async.AsyncResult(task_id)
    if task.ready():
        return jsonify({'status': 'completed', 'result': task.result})
    else:
        return jsonify({'status': 'processing'})
```

**3. 前端改为轮询模式**
```typescript
// src/components/OptionAnalyzer.tsx
async function analyzeStock(ticker: string) {
  setLoading(true);
  
  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ticker })
  });
  const data = await response.json();
  
  if (data.from_cache) {
    // 直接显示缓存结果
    setResult(data.data);
    setLoading(false);
  } else {
    // 轮询任务状态
    const taskId = data.task_id;
    const interval = setInterval(async () => {
      const statusRes = await fetch(`/api/task/${taskId}`);
      const status = await statusRes.json();
      
      if (status.status === 'completed') {
        clearInterval(interval);
        setResult(status.result);
        setLoading(false);
      }
    }, 1000); // 每秒查询一次
  }
}
```

#### 部署配置

**docker-compose.yml**
```yaml
version: '3.8'
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./dist:/usr/share/nginx/html
    depends_on:
      - flask

  flask:
    build: ./backend
    command: gunicorn -w 4 -b 0.0.0.0:5000 app:app
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/butterquant
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis

  celery_worker:
    build: ./backend
    command: celery -A tasks worker --loglevel=info --concurrency=4
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/butterquant
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - postgres

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: butterquant
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

**backend/Dockerfile**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt celery redis

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

**nginx.conf**
```nginx
events {
    worker_connections 1024;
}

http {
    upstream flask_backend {
        server flask:5000;
    }

    server {
        listen 80;
        
        # 前端静态文件
        location / {
            root /usr/share/nginx/html;
            try_files $uri $uri/ /index.html;
        }
        
        # API 代理
        location /api/ {
            proxy_pass http://flask_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 300s;
        }
    }
}
```

#### 成本估算（云服务商）
- **阿里云/腾讯云**：约 ¥300-500/月
  - 2核4G ECS × 1（运行所有容器）
  - Redis 1G × 1
  - PostgreSQL 20G × 1
- **预期性能**：支持 **100-200 并发用户**

---

### 🚀 方案 2：混合架构（推荐用于中长期）

**核心思路**：保留 Python 计算引擎，但用 **Node.js/Go** 做 API 层

#### 为什么这样做？
- Python 擅长科学计算（NumPy、Pandas），但不擅长高并发 I/O
- Node.js/Go 擅长高并发 API 处理，但科学计算库不如 Python 成熟
- **分工合作**：Node.js 处理 API + 缓存，Python 专注计算

#### 架构图
```
用户 → Cloudflare CDN → Node.js API (Express/Fastify)
                            ↓
                         Redis 缓存
                            ↓ (cache miss)
                         RabbitMQ 队列
                            ↓
                         Python Workers (Celery)
                            ↓
                         PostgreSQL
```

#### 技术栈
| 层级 | 技术 | 作用 |
|------|------|------|
| CDN | Cloudflare | 静态资源 + 边缘缓存 |
| API 层 | **Node.js (Fastify)** | 高并发 API 处理 |
| 计算层 | Python (Celery) | 量化计算 |
| 缓存 | Redis Cluster | 分布式缓存 |
| 数据库 | PostgreSQL + TimescaleDB | 时序数据优化 |
| 消息队列 | RabbitMQ | 任务分发 |

#### Node.js API 示例
```javascript
// api/server.js
const fastify = require('fastify')();
const redis = require('redis');
const amqp = require('amqplib');
const { v4: uuidv4 } = require('uuid');

const redisClient = redis.createClient();

fastify.post('/api/analyze', async (request, reply) => {
  const { ticker } = request.body;
  
  // 1. 检查缓存
  const cached = await redisClient.get(`analysis:${ticker}`);
  if (cached) {
    return { success: true, data: JSON.parse(cached), from_cache: true };
  }
  
  // 2. 发送到 RabbitMQ
  const connection = await amqp.connect('amqp://localhost');
  const channel = await connection.createChannel();
  await channel.assertQueue('analysis_tasks');
  
  const taskId = uuidv4();
  channel.sendToQueue('analysis_tasks', Buffer.from(JSON.stringify({
    task_id: taskId,
    ticker: ticker
  })));
  
  return { success: true, task_id: taskId, status: 'processing' };
});

fastify.listen({ port: 3000, host: '0.0.0.0' });
```

#### Python Worker (保持不变)
```python
# backend/worker.py
import pika
import json
from analyzer import ButterflyAnalyzer
import redis

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='analysis_tasks')

def callback(ch, method, properties, body):
    task = json.loads(body)
    ticker = task['ticker']
    
    analyzer = ButterflyAnalyzer(ticker)
    result = analyzer.full_analysis()
    
    # 存入 Redis + PostgreSQL
    redis_client.setex(f"analysis:{ticker}", 300, json.dumps(result))
    
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(queue='analysis_tasks', on_message_callback=callback)
channel.start_consuming()
```

#### 优势
- **性能**：Node.js 可处理 **1000+ 并发**
- **成本**：Python Worker 可按需扩展（Kubernetes HPA）
- **稳定性**：计算层崩溃不影响 API 层

#### 成本估算
- **阿里云 ACK (Kubernetes)**：约 ¥800-1200/月
  - Node.js Pod × 3（2核2G）
  - Python Worker Pod × 5（4核8G）
  - Redis Cluster × 3
  - PostgreSQL RDS

---

### ⚡ 方案 3：完全重写（长期方案）

**核心思路**：用 **Rust/Go** 重写计算引擎

#### 为什么？
- Python 的 GIL（全局解释器锁）限制了多核利用率
- Rust/Go 的并发性能是 Python 的 **10-100倍**
- 但开发成本高，需要重新实现所有量化模型

#### 技术栈
- **后端**：Rust (Actix-web) 或 Go (Gin)
- **计算库**：
  - Rust: `ndarray`, `polars`, `statrs`
  - Go: `gonum`, `gota`
- **部署**：单二进制文件，无需 Python 环境

#### 示例（Rust）
```rust
// src/analyzer.rs
use ndarray::Array1;
use statrs::distribution::Normal;

pub struct ButterflyAnalyzer {
    ticker: String,
    prices: Array1<f64>,
}

impl ButterflyAnalyzer {
    pub fn black_scholes(&self, s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
        let d1 = ((s / k).ln() + (r + 0.5 * sigma.powi(2)) * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        s * normal.cdf(d1) - k * (-r * t).exp() * normal.cdf(d2)
    }
}
```

#### 优势
- **性能**：单机支持 **10,000+ 并发**
- **成本**：1台 4核8G 服务器即可（¥200/月）

#### 劣势
- **开发周期**：3-6个月
- **维护成本**：需要 Rust/Go 专家

---

## 📈 推荐路线图

### 阶段 1：快速上线（1-2周）
✅ 采用 **方案1**
- 添加 Redis 缓存
- 引入 Celery 异步任务
- 部署到 Docker Compose

### 阶段 2：优化性能（1-2月）
✅ 采用 **方案2**
- Node.js API 层
- Kubernetes 部署
- 监控告警（Prometheus + Grafana）

### 阶段 3：终极优化（6月+）
✅ 评估 **方案3**
- 根据用户增长决定是否重写
- 如果日活 < 10,000，方案2 已足够

---

## 🔧 立即可做的优化（无需架构改动）

### 1. 启用 Gunicorn 多进程
```bash
# backend/Procfile
web: gunicorn -w 4 -k gevent --worker-connections 1000 app:app
```

### 2. 添加简单缓存
```python
# backend/app.py
import time

CACHE = {}
CACHE_TTL = 300  # 5分钟

@app.route('/api/analyze', methods=['POST'])
def analyze():
    ticker = request.json.get('ticker', 'AAPL').upper()
    
    # 检查缓存
    if ticker in CACHE:
        cached_time, cached_result = CACHE[ticker]
        if time.time() - cached_time < CACHE_TTL:
            return jsonify({'success': True, 'data': cached_result, 'from_cache': True})
    
    # 计算
    analyzer = ButterflyAnalyzer(ticker)
    result = analyzer.full_analysis()
    
    # 存入缓存
    CACHE[ticker] = (time.time(), result)
    
    return jsonify({'success': True, 'data': result})
```

### 3. 数据库连接池
```python
# backend/database.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:pass@localhost/butterquant',
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

---

## 💰 成本对比

| 方案 | 月成本 | 支持并发 | 开发周期 | 维护难度 |
|------|--------|----------|----------|----------|
| 当前架构 | ¥0 | 10 | - | 低 |
| 方案1 (Celery) | ¥300-500 | 100-200 | 1-2周 | 中 |
| 方案2 (Node.js) | ¥800-1200 | 1000+ | 1-2月 | 中高 |
| 方案3 (Rust) | ¥200-400 | 10000+ | 6月+ | 高 |

---

## 🎯 下一步行动

1. **确定目标用户规模**：预计日活多少？
2. **选择方案**：建议先做方案1，快速验证市场
3. **可以帮您**：
   - 生成完整的 Docker Compose 配置
   - 改造现有代码支持 Celery
   - 编写部署脚本

---

## 📝 总结

- **立即优化**：今天就能完成，性能提升 2-3倍
- **方案1**：1-2周上线，支持 100-200 并发，成本 ¥300-500/月
- **方案2**：1-2月完成，支持 1000+ 并发，成本 ¥800-1200/月
- **方案3**：6月+完成，支持 10000+ 并发，成本 ¥200-400/月

建议路径：**立即优化 → 方案1 → 根据用户增长决定是否升级到方案2**
