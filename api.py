import base64
import datetime
import logging
import os
import re
from pathlib import Path
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from app import FashionAIAgent, ImageAnalyzer, OutfitRecommender, ToolExecutor
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        'FASHIONAI_CORS_ORIGINS',
        'http://127.0.0.1:5173,http://localhost:5173',
    ).split(',')
    if origin.strip()
]
CORS(app, resources={r'/*': {'origins': allowed_origins}})
logger = logging.getLogger("FashionAI")

# 会话存储（每个用户独立）
sessions = {}

PRODUCT_FILE_PATH = Path(__file__).resolve().parent / 'knowledge' / 'product_info.txt'
ORDERS_FILE_PATH = Path(__file__).resolve().parent / 'data' / 'orders.json'
AFTER_SALES_FILE_PATH = Path(__file__).resolve().parent / 'data' / 'after_sales.json'

def load_products_from_file():
    """读取并解析商品信息文件（统一复用 app.py 的商品目录解析）"""
    items = ToolExecutor.list_products(force_reload=False)
    normalized = []

    for item in items:
        raw_category = str(item.get('category', '') or '')
        if '上衣' in raw_category:
            category = '上衣'
        elif '裤子' in raw_category:
            category = '裤子'
        else:
            category = raw_category or None

        price = item.get('price', '')
        price_text = f"￥{price}" if isinstance(price, (int, float)) else str(price or '')

        normalized.append({
            'gender': item.get('gender'),
            'category': category,
            'name': item.get('name', ''),
            'price': price_text,
            'image_url': item.get('image_url', ''),
            'shop_url': item.get('shop_url', ''),
            'color': item.get('color', ''),
            'size': item.get('size', ''),
            'material': item.get('material', ''),
            'features': item.get('selling_point', '') or item.get('desc', ''),
        })

    return normalized


def load_orders_from_file():
    """读取订单数据文件"""
    if not ORDERS_FILE_PATH.exists():
        return []

    try:
        data = json.loads(ORDERS_FILE_PATH.read_text(encoding='utf-8'))
    except Exception:
        return []

    orders = data.get('orders', []) if isinstance(data, dict) else []
    return orders if isinstance(orders, list) else []


def load_after_sales_from_file() -> list[dict]:
    """读取售后工单文件。"""
    if not AFTER_SALES_FILE_PATH.exists():
        return []

    try:
        data = json.loads(AFTER_SALES_FILE_PATH.read_text(encoding='utf-8'))
    except Exception:
        return []

    tickets = data.get('after_sales', []) if isinstance(data, dict) else []
    return tickets if isinstance(tickets, list) else []


def save_after_sales_to_file(tickets: list[dict]) -> None:
    """写回售后工单文件。"""
    payload = {'after_sales': tickets}
    AFTER_SALES_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    AFTER_SALES_FILE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _next_after_sales_id(tickets: list[dict]) -> str:
    today = datetime.datetime.now().strftime('%Y%m%d')
    prefix = f'AS{today}'
    max_seq = 0
    for ticket in tickets:
        current = str(ticket.get('after_sales_id', '')).strip()
        if not current.startswith(prefix):
            continue
        tail = current[len(prefix):]
        if tail.isdigit():
            max_seq = max(max_seq, int(tail))
    return f'{prefix}{max_seq + 1:04d}'


def _find_latest_after_sales_ticket(order_id: str, action: str | None = None) -> dict | None:
    normalized_order = str(order_id or '').strip().lower()
    if not normalized_order:
        return None

    normalized_action = str(action or '').strip().lower()
    candidates = []
    for ticket in load_after_sales_from_file():
        current_order_id = str(ticket.get('order_id', '')).strip().lower()
        current_action = str(ticket.get('action', '')).strip().lower()
        if current_order_id != normalized_order:
            continue
        if normalized_action and current_action != normalized_action:
            continue
        candidates.append(ticket)

    if not candidates:
        return None

    candidates.sort(key=lambda x: str(x.get('created_at', '')), reverse=True)
    return candidates[0]


def _find_after_sales_ticket_by_id(after_sales_id: str) -> dict | None:
    normalized = str(after_sales_id or '').strip().lower()
    if not normalized:
        return None

    for ticket in load_after_sales_from_file():
        current_id = str(ticket.get('after_sales_id', '')).strip().lower()
        if current_id == normalized:
            return ticket
    return None


def _create_after_sales_ticket(order: dict, issue_type: str, action: str, user_message: str) -> tuple[dict, bool]:
    tickets = load_after_sales_from_file()
    order_id = str(order.get('order_id', '')).strip()
    active_statuses = {'submitted', 'reviewing', 'approved'}

    existing = _find_latest_after_sales_ticket(order_id, action=action)
    if isinstance(existing, dict) and str(existing.get('status', '')).strip().lower() in active_statuses:
        return existing, False

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ticket = {
        'after_sales_id': _next_after_sales_id(tickets),
        'order_id': order_id,
        'action': action,
        'issue_type': issue_type,
        'status': 'submitted',
        'created_at': now,
        'updated_at': now,
        'remark': user_message,
    }

    tickets.append(ticket)
    save_after_sales_to_file(tickets)
    return ticket, True


def _is_after_sales_status_query(message: str) -> bool:
    text = _normalize_text(message)
    keywords = [
        '退了吗', '退到哪了', '进度', '状态', '处理到哪', '处理好了吗',
        '什么时候退款', '多久到账', '退款去哪', '退款到哪里', '售后进度'
    ]
    return any(k in text for k in keywords)


def _map_after_sales_status(status: str) -> str:
    mapping = {
        'submitted': '已提交',
        'reviewing': '审核中',
        'approved': '已通过',
        'rejected': '已驳回',
        'refunded': '已退款',
        'closed': '已关闭',
    }
    return mapping.get(str(status or '').strip().lower(), '处理中')


def _refund_target_text(order: dict) -> str:
    pay_method = str(order.get('payment_method', '')).strip()
    if '微信' in pay_method:
        return '原微信支付账户'
    if '支付宝' in pay_method:
        return '原支付宝账户'
    if pay_method:
        return f'原支付渠道（{pay_method}）'
    return '原支付账户'


def _format_after_sales_submit_reply(order: dict, ticket: dict) -> str:
    status_text = _map_after_sales_status(ticket.get('status', 'submitted'))
    action_text = '退款' if str(ticket.get('action', '')).strip().lower() == 'refund' else '换货'
    return (
        f"我已经为您创建了{action_text}申请，工单号：{ticket.get('after_sales_id', '未知')}。\n"
        f"当前状态：{status_text}。"
        f"若审核通过，款项将按原路退回至{_refund_target_text(order)}，通常1-3个工作日到账。"
    )


def _format_after_sales_status_reply(order: dict, ticket: dict) -> str:
    status_text = _map_after_sales_status(ticket.get('status', ''))
    action_text = '退款' if str(ticket.get('action', '')).strip().lower() == 'refund' else '换货'
    if str(ticket.get('status', '')).strip().lower() == 'refunded':
        return (
            f"您的{action_text}工单（{ticket.get('after_sales_id', '未知')}）状态：{status_text}。\n"
            f"款项已退回至{_refund_target_text(order)}，请您留意到账通知。"
        )
    return (
        f"您的{action_text}工单（{ticket.get('after_sales_id', '未知')}）当前状态：{status_text}。\n"
        f"审核通过后会原路退回至{_refund_target_text(order)}，一般1-3个工作日到账。"
    )


def _update_after_sales_status(after_sales_id: str, status: str, remark: str = '') -> dict | None:
    valid_status = {'submitted', 'reviewing', 'approved', 'rejected', 'refunded', 'closed'}
    normalized_status = str(status or '').strip().lower()
    if normalized_status not in valid_status:
        return None

    tickets = load_after_sales_from_file()
    normalized_id = str(after_sales_id or '').strip().lower()
    for ticket in tickets:
        current_id = str(ticket.get('after_sales_id', '')).strip().lower()
        if current_id != normalized_id:
            continue

        ticket['status'] = normalized_status
        ticket['updated_at'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if remark:
            ticket['remark'] = remark
        save_after_sales_to_file(tickets)
        return ticket

    return None


def _is_order_query_intent(message: str) -> bool:
    text = _normalize_text(message)
    keywords = ['订单查询', '查订单', '查询订单', '订单', '物流', '快递单号']
    return any(k in text for k in keywords)


def _is_coupon_intent(message: str) -> bool:
    text = _normalize_text(message)
    keywords = ['优惠券', '优惠卷', '券', '折扣', '满减', '打折', '有优惠吗', '有活动吗']
    return any(k in text for k in keywords)


def _parse_coupon_level(message: str) -> str:
    text = _normalize_text(message)
    if any(k in text for k in ['新用户', '新人', '首次', '第一次']):
        return '新用户'
    if any(k in text for k in ['vip', '会员', '黑卡', '高级会员']):
        return 'VIP'
    return '普通'


def _format_coupon_reply(coupons: list[dict], level: str) -> str:
    if not coupons:
        return '当前没有可领取的优惠券，您告诉我预算和想买的品类，我可以再帮您找更划算的款。'

    level_text = '新用户' if level == '新用户' else ('VIP' if level == 'VIP' else '当前账号')
    lines = [f'有的，给您查到了 {level_text} 可用优惠券：', '']
    for idx, item in enumerate(coupons, 1):
        name = str(item.get('name', f'优惠券{idx}') or f'优惠券{idx}')
        condition = str(item.get('condition', '') or '').strip()
        if condition:
            lines.append(f'{idx}. {name}（{condition}）')
        else:
            lines.append(f'{idx}. {name}')
    lines.append('')
    lines.append('您现在要买哪一款？我可以顺手帮您按券后更划算的方式推荐。')
    return '\n'.join(lines).strip()


def _handle_coupon_workflow(message: str) -> str | None:
    if not _is_coupon_intent(message):
        return None

    level = _parse_coupon_level(message)
    raw = ToolExecutor.get_coupon(level=level)
    try:
        payload = json.loads(raw)
    except Exception:
        payload = {'success': False, 'coupons': []}

    if not isinstance(payload, dict) or not payload.get('success'):
        return '我这边暂时没查到优惠券信息，您可以稍后再试，我也能先帮您按预算筛选高性价比款式。'

    coupons = payload.get('coupons', []) if isinstance(payload.get('coupons', []), list) else []
    return _format_coupon_reply(coupons, level)


def _extract_phone(message: str) -> str:
    match = re.search(r'(1[3-9]\d{9})', message or '')
    return match.group(1) if match else ''


def _extract_name(message: str) -> str:
    text = (message or '').strip()
    if not text:
        return ''

    # 过滤明显的业务意图词，避免把“订单查询/查订单/物流状态”等误识别为姓名
    normalized_text = re.sub(r'\s+', '', text)
    blocked_name_tokens = {
        '订单', '查订单', '订单查询', '查询订单', '物流', '物流查询', '快递', '快递单号',
        '售后', '退货', '换货', '退款', '查询', '订单状态', '我想查询订单', '我想查订单',
        '想查询订单', '想查订单', '查一下订单', '帮我查订单', '我要查订单'
    }
    if normalized_text in blocked_name_tokens:
        return ''

    explicit = re.search(r'(?:姓名|名字)[:：]?\s*([\u4e00-\u9fa5]{2,8})', text)
    if explicit:
        return explicit.group(1)

    intro = re.search(r'我(?:叫|是)\s*([\u4e00-\u9fa5]{2,8})', text)
    if intro:
        return intro.group(1)

    # 兼容“13800138005林晓”或“13800138005 林晓”这类手机号+姓名拼接输入
    attached = re.search(r'(?:1[3-9]\d{9})\s*([\u4e00-\u9fa5]{2,8})$', text)
    if attached:
        return attached.group(1)

    cleaned = re.sub(r'\s+', '', text)
    if any(token in cleaned for token in ['订单', '物流', '快递', '售后', '退货', '换货', '退款']):
        return ''
    if re.fullmatch(r'[\u4e00-\u9fa5]{2,8}', cleaned):
        if cleaned in blocked_name_tokens:
            return ''
        return cleaned

    return ''


def _find_orders_by_phone_name(phone: str, name: str) -> list[dict]:
    orders = load_orders_from_file()
    matched = [
        order for order in orders
        if str(order.get('phone', '')).strip() == phone and str(order.get('name', '')).strip() == name
    ]
    matched.sort(key=lambda x: str(x.get('order_time', '')), reverse=True)
    return matched


def _find_orders_by_phone(phone: str) -> list[dict]:
    orders = load_orders_from_file()
    matched = [
        order for order in orders
        if str(order.get('phone', '')).strip() == phone
    ]
    matched.sort(key=lambda x: str(x.get('order_time', '')), reverse=True)
    return matched


def _find_order_by_tracking_number(tracking_number: str) -> dict | None:
    normalized = str(tracking_number or '').strip().lower()
    if not normalized:
        return None

    orders = load_orders_from_file()
    matched = []
    for order in orders:
        shipping = order.get('shipping', {}) or {}
        tracking = str(shipping.get('tracking_number', '')).strip().lower()
        if tracking == normalized:
            matched.append(order)

    if not matched:
        return None

    matched.sort(key=lambda x: str(x.get('order_time', '')), reverse=True)
    return matched[0]


def _find_order_by_order_id(order_id: str) -> dict | None:
    normalized = str(order_id or '').strip().lower()
    if not normalized:
        return None

    orders = load_orders_from_file()
    matched = []
    for order in orders:
        current_id = str(order.get('order_id', '')).strip().lower()
        if current_id == normalized:
            matched.append(order)

    if not matched:
        return None

    matched.sort(key=lambda x: str(x.get('order_time', '')), reverse=True)
    return matched[0]


def _remember_order_context(agent, order: dict):
    agent._last_order_context = order
    agent._active_order_context = order


def _build_order_status_followup(status: str) -> str:
    status_text = str(status or '')
    if '配送中' in status_text or '派送中' in status_text:
        return '您的订单正在派送中，需要我帮您催单或查询实时位置吗？'
    if '待发货' in status_text:
        return '商品正在准备中，需要帮您催一下发货吗？'
    if '已签收' in status_text:
        return '商品已签收，您还满意吗？需要申请售后吗？'
    if '待付款' in status_text:
        return '您的订单还未付款，需要帮您提醒付款吗？'
    return '我已看到您的订单信息，需要我继续帮您跟进物流或售后吗？'


def _refresh_order_followup_state(agent, order: dict, assistant_response: str) -> None:
    """根据当前订单回复内容，刷新订单追问状态。"""
    text = str(assistant_response or '')
    shipping = order.get('shipping', {}) or {}
    status = str(shipping.get('status', '') or '')

    needs_expedite_confirm = ('待发货' in status) and ('催一下发货' in text)
    agent._pending_order_expedite = bool(needs_expedite_confirm)


def _handle_order_expedite_followup(agent, message: str) -> str | None:
    active_order = getattr(agent, '_active_order_context', None)
    if not isinstance(active_order, dict):
        return None

    text = _normalize_text(message)
    pending_expedite = bool(getattr(agent, '_pending_order_expedite', False))

    if pending_expedite:
        want_expedite = _is_affirmative_reply(text) or any(k in text for k in ['需要', '要', '帮我催', '催一下', '催发货', '帮我催发货'])
        if want_expedite:
            agent._pending_order_expedite = False
            shipping = active_order.get('shipping', {}) or {}
            return (
                f"已为您提交催发货提醒。订单号：{active_order.get('order_id', '未知')}。"
                f"当前状态：{shipping.get('status', '未知')}，快递单号：{shipping.get('tracking_number', '未知')}。"
                "商家发货后我会继续帮您跟进物流。"
            )

        reject_tokens = ['不用', '不需要', '先不用', '不用了', '暂时不用']
        if any(k in text for k in reject_tokens):
            agent._pending_order_expedite = False
            return '好的，先不催发货。您如果改变主意，随时告诉我“帮我催发货”就行。'

    why_not_found_tokens = ['为什么没找到', '怎么没找到', '没找到', '查不到']
    if any(k in text for k in why_not_found_tokens):
        shipping = active_order.get('shipping', {}) or {}
        return (
            f"我这边能查到您的订单：{active_order.get('order_id', '未知')}。"
            f"当前物流状态是“{shipping.get('status', '未知')}”，快递单号 {shipping.get('tracking_number', '未知')}。"
            "如果您要，我现在就继续帮您催发货。"
        )

    return None


def _extract_tracking_number(message: str) -> str:
    text = str(message or '')
    match = re.search(r'快递单号\s*[：:]\s*([A-Za-z0-9\-]+)', text)
    return match.group(1).strip() if match else ''


def _extract_tracking_number_fallback(message: str) -> str:
    text = str(message or '').strip()
    if not text:
        return ''

    explicit = _extract_tracking_number(text)
    if explicit:
        return explicit

    # 兜底识别：纯单号输入或在句中出现形如 SF123456789012 的快递号
    match = re.search(r'\b([A-Za-z]{1,4}\d{8,20})\b', text)
    return match.group(1).strip() if match else ''


def _extract_order_id(message: str) -> str:
    text = str(message or '').strip()
    if not text:
        return ''

    explicit = re.search(r'订单号\s*[：:]\s*([A-Za-z0-9\-]+)', text)
    if explicit:
        return explicit.group(1).strip()

    fallback = re.search(r'\b(ORD[0-9A-Za-z\-]{4,})\b', text, re.IGNORECASE)
    return fallback.group(1).strip() if fallback else ''


def _build_order_context_inline(order: dict) -> str:
    shipping = order.get('shipping', {}) or {}
    return (
        f"订单号：{order.get('order_id', '未知')}；"
        f"快递单号：{shipping.get('tracking_number', '未知')}；"
        f"物流状态：{shipping.get('status', '未知')}；"
        f"快递公司：{shipping.get('company', '未知')}"
    )


def _sync_order_context_from_message(agent, message: str):
    text = str(message or '').strip()
    if not text:
        return

    tracking = _extract_tracking_number_fallback(text)
    if tracking:
        order = _find_order_by_tracking_number(tracking)
        if isinstance(order, dict):
            _remember_order_context(agent, order)
            return

    order_id = _extract_order_id(text)
    if order_id:
        order = _find_order_by_order_id(order_id)
        if isinstance(order, dict):
            _remember_order_context(agent, order)


def _is_after_sales_intent(message: str) -> bool:
    text = _normalize_text(message)
    keywords = [
        '货不对版', '货不对板', '不对版', '不对板', '不是这个', '发错', '错发', '质量', '破损', '瑕疵',
        '色差', '尺寸不对', '尺码不对', '不合身', '退货', '换货', '退款', '售后',
        '投诉', '问题件', '有问题', '坏了', '开线', '掉色', '异味',
        '退了吗', '进度', '状态', '多久到账', '退款去哪', '处理到哪'
    ]
    return any(k in text for k in keywords)


def _detect_issue_type(message: str) -> str:
    text = _normalize_text(message)
    if any(k in text for k in ['货不对版', '货不对板', '不对版', '不对板', '发错', '错发', '不是这个']):
        return 'wrong_item'
    if any(k in text for k in ['尺寸不对', '尺码不对', '不合身']):
        return 'size_issue'
    if any(k in text for k in ['色差', '颜色不对']):
        return 'color_issue'
    if any(k in text for k in ['质量', '破损', '瑕疵', '坏了', '开线', '掉色', '异味']):
        return 'quality_issue'
    if any(k in text for k in ['退货', '换货', '退款', '售后']):
        return 'after_sales'
    return 'generic'


def _detect_after_sales_action(message: str) -> str:
    text = _normalize_text(message)
    has_refund = any(k in text for k in ['退款', '仅退款', '我要退款'])
    has_return = any(k in text for k in ['退货', '我要退货'])
    has_exchange = any(k in text for k in ['换货', '换一个', '换新'])

    if '退货退款' in text or (has_return and has_refund):
        return 'refund'
    if has_refund and not has_exchange:
        return 'refund'
    if has_return and not has_exchange:
        return 'refund'
    if has_exchange and not has_refund:
        return 'exchange'
    if any(k in text for k in ['继续物流', '继续跟进物流', '跟进物流', '催单', '催一下']):
        return 'logistics'
    return 'none'


def _build_after_sales_ai_prompt(
    user_message: str,
    order: dict,
    issue_type: str,
    action: str,
    ticket: dict | None = None,
    allow_submit_claim: bool = True,
) -> str:
    shipping = order.get('shipping', {}) or {}
    items = order.get('items', []) or []
    first_item = items[0].get('name', '未知商品') if items else '未知商品'

    ticket_text = '无'
    if isinstance(ticket, dict):
        ticket_text = (
            f"工单号：{ticket.get('after_sales_id', '未知')}；"
            f"动作：{ticket.get('action', '未知')}；"
            f"状态：{_map_after_sales_status(ticket.get('status', ''))}"
        )

    submit_rule = (
        '5) 未创建工单时，严禁使用“已提交/已申请成功”等表述。\n'
        if not allow_submit_claim else
        '5) 仅在明确存在工单号时，才能说“已提交/已申请成功”。\n'
    )

    return f"""你是电商售后客服，请根据上下文像真人一样回复用户。

【硬性要求】
1) 用户已提供并确认订单，不要再索要订单号或快递单号。
2) 先同理用户，再给明确下一步。
3) 若用户已明确动作（如退款/退货退款/换货），直接进入该流程，不再反问“您是要退款还是换货”。
4) 回复简洁自然，控制在2-4句。
{submit_rule}6) 若有工单，必须带上工单号和当前状态。

【当前订单】
- 订单号：{order.get('order_id', '未知')}
- 快递单号：{shipping.get('tracking_number', '未知')}
- 物流状态：{shipping.get('status', '未知')}
- 快递公司：{shipping.get('company', '未知')}
- 商品：{first_item}

【识别结果】
- 问题类型：{issue_type}
- 用户动作倾向：{action}
- 当前工单：{ticket_text}

【用户原话】
{user_message}

请直接输出给用户的话，不要输出分析过程。"""


def _generate_after_sales_ai_response(
    agent,
    user_message: str,
    order: dict,
    issue_type: str,
    action: str,
    user_info: dict,
    ticket: dict | None = None,
    allow_submit_claim: bool = True,
) -> str:
    prompt = _build_after_sales_ai_prompt(
        user_message,
        order,
        issue_type,
        action,
        ticket=ticket,
        allow_submit_claim=allow_submit_claim,
    )
    return agent.chat(prompt, user_info)


def _handle_after_sales_workflow(agent, message: str, user_info: dict) -> tuple[str, dict] | None:
    result = _try_handle_order_after_sales(agent, message)
    if result is None:
        return None

    order_context, issue_type, action = result
    order_id = str(order_context.get('order_id', '')).strip()
    latest_ticket = _find_latest_after_sales_ticket(order_id)

    if _is_after_sales_status_query(message):
        if isinstance(latest_ticket, dict):
            return _format_after_sales_status_reply(order_context, latest_ticket), order_context
        return '我还没有查到该订单的售后工单记录。若您需要，我可以现在帮您提交退款申请。', order_context

    if action in {'refund', 'exchange'}:
        ticket, created = _create_after_sales_ticket(order_context, issue_type, action, message)
        if created:
            return _format_after_sales_submit_reply(order_context, ticket), order_context
        return _format_after_sales_status_reply(order_context, ticket), order_context

    response = _generate_after_sales_ai_response(
        agent,
        message,
        order_context,
        issue_type,
        action,
        user_info,
        ticket=latest_ticket,
        allow_submit_claim=False,
    )
    return response, order_context


def _try_handle_order_after_sales(agent, message: str) -> tuple[dict, str, str] | None:
    active_order = getattr(agent, '_active_order_context', None)
    if not isinstance(active_order, dict):
        return None

    if not _is_after_sales_intent(message):
        if _wants_after_sales_followup(agent, message):
            return active_order, 'after_sales', 'none'
        return None

    issue_type = _detect_issue_type(message)
    action = _detect_after_sales_action(message)
    return active_order, issue_type, action


def _try_handle_order_ai_assistant(agent, message: str) -> tuple[str, dict] | None:
    text = str(message or '').strip()
    if not text:
        return None

    is_trigger = ('我的订单' in text and '快递单号' in text) or text.startswith('我的订单【快递单号')
    if not is_trigger:
        return None

    tracking = _extract_tracking_number(text)
    active_order = getattr(agent, '_active_order_context', None)
    last_order = getattr(agent, '_last_order_context', None)

    selected_order = None
    if tracking:
        if isinstance(active_order, dict):
            active_tracking = str((active_order.get('shipping', {}) or {}).get('tracking_number', '')).strip()
            if active_tracking == tracking:
                selected_order = active_order

        if selected_order is None and isinstance(last_order, dict):
            last_tracking = str((last_order.get('shipping', {}) or {}).get('tracking_number', '')).strip()
            if last_tracking == tracking:
                selected_order = last_order

        if selected_order is None:
            selected_order = _find_order_by_tracking_number(tracking)
    else:
        if isinstance(active_order, dict):
            selected_order = active_order
        elif isinstance(last_order, dict):
            selected_order = last_order

    if not isinstance(selected_order, dict):
        return None

    _remember_order_context(agent, selected_order)
    shipping = selected_order.get('shipping', {}) or {}
    response = _build_order_status_followup(str(shipping.get('status', '未知')))
    _refresh_order_followup_state(agent, selected_order, response)
    return response, selected_order


def _format_order_detail_text(order: dict, total_matches: int = 1) -> str:
    shipping = order.get('shipping', {}) or {}
    items = order.get('items', []) or []
    item_lines = []
    for item in items:
        item_lines.append(f"- {item.get('name', '未知商品')} x{item.get('quantity', 1)}，¥{item.get('price', 0)}")

    tracking = shipping.get('tracking_history', []) or []
    track_lines = []
    for t in tracking[:5]:
        track_lines.append(f"- {t.get('time', '')} {t.get('status', '')}")

    latest_tip = ''
    if total_matches > 1:
        latest_tip = f"\n（已为您返回最新一笔订单，共匹配到 {total_matches} 笔）\n"

    return (
        f"📦 订单查询结果{latest_tip}"
        f"订单号：{order.get('order_id', '未知')}\n"
        f"下单时间：{order.get('order_time', '未知')}\n"
        f"商品清单：\n{chr(10).join(item_lines) if item_lines else '- 暂无'}\n"
        f"总金额：¥{order.get('total_amount', 0)}\n"
        f"快递公司：{shipping.get('company', '未知')}\n"
        f"快递单号：{shipping.get('tracking_number', '未知')}\n"
        f"物流状态：{shipping.get('status', '未知')}\n"
        f"物流轨迹：\n{chr(10).join(track_lines) if track_lines else '- 暂无物流轨迹'}\n"
        f"优惠券：{order.get('coupon_used', '无')}\n"
        f"支付方式：{order.get('payment_method', '未知')}"
    )


def _handle_order_query_dialog(agent, message: str) -> str | None:
    is_intent = _is_order_query_intent(message)
    pending = bool(getattr(agent, '_pending_order_query', False))

    if not is_intent and not pending:
        return None

    form = getattr(agent, '_order_query_form', None)
    if not isinstance(form, dict):
        form = {'phone': '', 'name': ''}

    msg_phone = _extract_phone(message)
    msg_name = _extract_name(message)

    # 输入了新手机号但没带姓名时，清理旧姓名，避免“姓名串线”
    old_phone = str(form.get('phone', '')).strip()
    if msg_phone and msg_phone != old_phone and not msg_name:
        form['name'] = ''

    if msg_phone:
        form['phone'] = msg_phone
    if msg_name:
        form['name'] = msg_name

    agent._order_query_form = form

    missing = []
    if not form.get('phone'):
        missing.append('手机号')
    if not form.get('name'):
        missing.append('下单姓名')

    if missing:
        agent._pending_order_query = True
        if len(missing) == 2:
            return "可以的，我来帮您查订单。请提供：手机号和下单姓名。"

        # 保持统一引导口径，同时明确已识别的信息，避免用户误以为系统未理解输入
        if missing[0] == '下单姓名':
            return f"已收到手机号 {form.get('phone', '')}。可以的，我来帮您查订单。请提供：手机号和下单姓名。"
        return f"已收到下单姓名“{form.get('name', '')}”。可以的，我来帮您查订单。请提供：手机号和下单姓名。"

    matched = _find_orders_by_phone_name(form['phone'], form['name'])
    agent._pending_order_query = False
    agent._order_query_form = {'phone': '', 'name': ''}

    if not matched:
        # 如果手机号存在但姓名不匹配，给出更明确指引，避免用户重复输入却无反馈
        phone_matches = _find_orders_by_phone(form['phone'])
        if phone_matches:
            candidate_names = []
            for order in phone_matches:
                candidate = str(order.get('name', '')).strip()
                if candidate and candidate not in candidate_names:
                    candidate_names.append(candidate)

            if candidate_names:
                name_hint = '、'.join(candidate_names[:3])
                return f"我查到手机号 {form['phone']} 下有订单，但姓名与“{form['name']}”不一致。请确认下单姓名（例如：{name_hint}）。"

        # 没有匹配到时清空活跃订单上下文，避免后续售后流程误用旧订单
        if hasattr(agent, '_active_order_context'):
            agent._active_order_context = None
        return f"未找到订单（手机号：{form['phone']}，姓名：{form['name']}），请检查后重试，或直接提供订单号。"

    _remember_order_context(agent, matched[0])
    return _format_order_detail_text(matched[0], total_matches=len(matched))


def filter_query_value(value: str, mapping: dict):
    if not value:
        return value
    value = value.strip()
    return mapping.get(value, value)


def _normalize_text(value: str) -> str:
    return (value or '').strip().lower()


def _wants_outfit_generation(message: str) -> bool:
    text = _normalize_text(message)
    keywords = [
        '生成合成图', '合成图', '搭配图', '效果图', '穿搭图', '出图', '生成图片', '做图', '搭配生成',
        '帮我做一版', '帮我生成', '做一张', '做一版', '出一张', '出一版', '看看效果', '看效果',
        '搭配看看', '生成看看', '合成一下', '顺手出图', '想看效果', '给我生成'
    ]
    if any(keyword in text for keyword in keywords):
        return True
    return bool(re.search(r'有.*(搭配|上身).*(效果图|效果|图片)|((搭配|上身).*(效果图|图片).*(吗|没|有))', text))


def _is_affirmative_reply(message: str) -> bool:
    text = _normalize_text(message)
    if text in {'要', '可以', '好', '行', '嗯', '嗯嗯', '好的', '可以的', '来吧', '可以啊', '行啊', '试试', '看看', '来一个', '可以呀'}:
        return True
    return bool(re.fullmatch(r'(继续|来吧|来个|来一版|发我|发我看看|发我看下|看一下|看看|可以|行|好)', text))


def _is_asset_request(message: str) -> bool:
    text = _normalize_text(message)
    keywords = [
        '图片', '看图', '发图', '链接', '购买链接', '商品链接', '详情链接', '给我链接',
        '这款链接', '这件链接', '第一款链接', '上衣链接', '图和链接'
    ]
    return any(k in text for k in keywords)


def _get_last_assistant_message(agent) -> str:
    history = getattr(agent, '_conversation_history', []) or []
    for item in reversed(history):
        if item.get('role') == 'assistant':
            return str(item.get('content', '') or '')
    return ''


def _wants_outfit_followup(agent, message: str) -> bool:
    """上下文意图识别：不仅看关键词，还结合上一轮助手内容判断是否在延续穿搭任务。"""
    if not agent.has_last_image_context():
        return False

    text = _normalize_text(message)
    if _wants_outfit_generation(text) or _is_affirmative_reply(text):
        return True

    if not _is_asset_request(text):
        return False

    last_assistant = _get_last_assistant_message(agent)
    last_context_hits = ['搭配', '效果图', '穿搭', '上衣', '下装', '上传了一张服装图片']
    return any(token in last_assistant for token in last_context_hits)


def _is_brief_followup(message: str) -> bool:
    text = _normalize_text(message)
    if not text:
        return False
    return len(re.sub(r'\s+', '', text)) <= 22


def _has_size_profile_clues(message: str) -> bool:
    text = _normalize_text(message)
    if not text:
        return False
    has_gender = any(k in text for k in ['男', '女', '男生', '女生', '先生', '女士'])
    has_height = bool(re.search(r'\b(1[2-9]\d|2[0-1]\d)\s*(cm|厘米)?\b', text))
    has_weight = bool(re.search(r'\b([3-9]\d|1\d\d)\s*(kg|公斤|斤)?\b', text))
    two_numbers = len(re.findall(r'\d{2,3}', text)) >= 2
    return has_gender or has_height or has_weight or two_numbers


def _wants_size_followup(agent, message: str) -> bool:
    text = _normalize_text(message)
    size_keywords = ['尺码', '穿什么码', '多大码', '推荐码', '码数', 'size']
    if any(k in text for k in size_keywords):
        return True

    pending_size = bool(getattr(agent, '_pending_size_recommendation', False))
    if pending_size and (_has_size_profile_clues(text) or _is_affirmative_reply(text)):
        return True

    last_assistant = _get_last_assistant_message(agent)
    if _has_size_profile_clues(text) and any(k in last_assistant for k in ['尺码', '身高', '体重', '男装尺码表', '女装尺码表']):
        return True

    return False


def _wants_after_sales_followup(agent, message: str) -> bool:
    text = _normalize_text(message)
    active_order = getattr(agent, '_active_order_context', None)
    if not isinstance(active_order, dict):
        return False

    if _is_after_sales_intent(text):
        return True

    followup_keywords = [
        '怎么办', '怎么处理', '然后呢', '那现在呢', '可以退吗', '能退吗', '怎么退', '怎么换',
        '退款吧', '换货吧', '退了没', '进度呢', '多久到账', '什么时候退款', '催一下', '帮我催'
    ]
    if any(k in text for k in followup_keywords):
        return True

    context_commit_keywords = [
        '不是看到我的订单', '不是查到订单了', '都看到订单了', '还不知道我买什么',
        '你不知道我买什么', '我买什么你不知道吗', '不是已经查到了吗'
    ]
    if any(k in text for k in context_commit_keywords):
        return True

    last_assistant = _get_last_assistant_message(agent)
    if _is_brief_followup(text) and any(k in last_assistant for k in ['工单', '退款', '换货', '售后', '物流状态', '订单号']):
        return True

    return False


def _wants_order_context_followup(agent, message: str) -> bool:
    text = _normalize_text(message)
    active_order = getattr(agent, '_active_order_context', None)
    if not isinstance(active_order, dict):
        return False

    pending_expedite = bool(getattr(agent, '_pending_order_expedite', False))
    if pending_expedite and (_is_affirmative_reply(text) or any(k in text for k in ['需要', '要', '帮我催', '催一下', '催发货', '帮我催发货'])):
        return True

    if _is_order_query_intent(text):
        return True

    logistics_keywords = [
        '物流', '快递', '单号', '订单状态', '状态', '到哪了', '什么时候到', '什么时候能到', '多久到', '几天到',
        '还没到', '还没收到', '没收到', '催单', '催一下', '查一下', '为什么没找到', '没找到', '查不到'
    ]
    if any(k in text for k in logistics_keywords):
        return True

    last_assistant = _get_last_assistant_message(agent)
    if _is_brief_followup(text) and any(k in last_assistant for k in ['订单号', '快递单号', '物流状态']):
        return True

    return False


def _resolve_dialog_intents(agent, message: str) -> dict:
    """统一意图层：结合当前输入和上下文，输出路由意图。"""
    text = _normalize_text(message)
    return {
        'coupon': _is_coupon_intent(text),
        'order_query': _is_order_query_intent(text),
        'order_followup': _wants_order_context_followup(agent, text),
        'after_sales': _is_after_sales_intent(text),
        'after_sales_followup': _wants_after_sales_followup(agent, text),
        'size_followup': _wants_size_followup(agent, text),
        'outfit_followup': _wants_outfit_followup(agent, text),
    }


def _should_bypass_order_query_form(agent, message: str) -> bool:
    """已有订单上下文时，避免订单表单抢占售后/上下文追问。"""
    if bool(getattr(agent, '_pending_order_query', False)):
        return False

    active_order = getattr(agent, '_active_order_context', None)
    if not isinstance(active_order, dict):
        return False

    text = _normalize_text(message)
    if _wants_after_sales_followup(agent, text) or _is_after_sales_intent(text):
        return True

    context_complaint_keywords = [
        '不是看到我的订单', '不是查到订单了', '还要我提供', '还要手机号',
        '我不是刚给过', '你不是知道我买什么'
    ]
    return any(k in text for k in context_complaint_keywords)


def _build_size_context_prompt(message: str) -> str:
    return (
        f"{message}\n"
        "[上下文路由提示] 用户正在继续尺码咨询，请优先沿尺码流程处理："
        "缺信息先追问，信息完整则给推荐。"
    )


def _handle_order_context_followup(agent, message: str) -> str | None:
    active_order = getattr(agent, '_active_order_context', None)
    if not isinstance(active_order, dict):
        return None

    shipping = active_order.get('shipping', {}) or {}
    status = str(shipping.get('status', '未知') or '未知')
    company = str(shipping.get('company', '未知') or '未知')
    tracking_number = str(shipping.get('tracking_number', '未知') or '未知')
    return (
        f"这笔订单当前物流状态：{status}。"
        f"快递公司：{company}，快递单号：{tracking_number}。"
        "如果您需要，我也可以继续帮您处理退款或换货。"
    )


def _build_image_advice_text(analysis: dict, user_info: dict) -> str:
    category = analysis.get('category', '服装')
    color = analysis.get('color', '未知')
    style = analysis.get('fashion_style') if analysis.get('fashion_style') != '未知' else analysis.get('style', '未知')
    scene = analysis.get('scene', '未知')
    material = analysis.get('material', '未知')
    suggestion = analysis.get('suggestion', '')

    category_tip = ""
    if '裤' in str(category):
        category_tip = (
            "- 这类卡其色裤子很适合搭配白色、米色、浅蓝这类干净的上衣，视觉会更清爽。\n"
            "- 如果想显腿长，可以选短款上衣或把上衣下摆稍微收一下；鞋子优先白鞋、板鞋或乐福鞋。\n"
            "- 日常/工作都能穿，工作场景建议把上装换成更利落的衬衫或轻薄夹克。"
        )
    elif any(k in str(category) for k in ['上衣', '衬衫', 'T恤', '卫衣', '外套']):
        category_tip = (
            "- 这类上装建议优先搭配纯色下装，避免整体太花。\n"
            "- 如果想更显精神，可以用同色系或浅深对比来做层次。\n"
            "- 若要通勤，外搭一件简洁外套会更稳；若偏休闲，配牛仔裤或卡其裤更容易出效果。"
        )
    else:
        category_tip = (
            "- 建议先围绕这件单品做简洁搭配，主色控制在2-3种以内，会更耐看。\n"
            "- 如果想更有层次，可以用同色系或一深一浅的方式去搭。"
        )

    if not suggestion:
        suggestion = '建议优先选择和单品风格一致的基础款进行搭配，这样更容易穿出整体感。'

    return f"""我先帮您拆解一下这件{color}{category}，再给您搭配建议：

🎨 **分析结论**：
- 类型：{category}
- 颜色：{color}
- 风格：{style}
- 场合：{scene}
- 材质：{material}

💡 **搭配建议**：
- {suggestion}
{category_tip}
- 这件单品更适合{scene}场景，整体搭配建议保持简洁、干净、有一点层次感。

如果您想看搭配效果图，我可以继续帮您做一版；如果先不看图，我也可以继续帮您细化搭配建议。"""


def _build_outfit_followup_hint() -> str:
    """生成自然的后续引导，不强制用户使用固定口令。"""
    return "如果您想看搭配效果图，我可以继续帮您做一版；如果先不看图，我也可以继续帮您细化搭配建议。"


def _append_outfit_followup_hint(text: str) -> str:
    base = (text or "").strip()
    hint = _build_outfit_followup_hint()
    if not base:
        return hint
    if hint in base:
        return base
    return f"{base}\n\n{hint}"


def _stream_text_response(text: str, extra_payload: dict | None = None):
    def send_event(payload: dict):
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def generate():
        chunk_size = 1
        for i in range(0, len(text), chunk_size):
            yield send_event({'chunk': text[i:i+chunk_size], 'done': False})
        if extra_payload:
            yield send_event({**extra_payload, 'done': False})
        yield send_event({'done': True})

    return generate


def _stream_outfit_recommendation(agent, image_base64: str, mime_type: str, user_info: dict, extra_payload: dict | None = None):
    def send_event(payload: dict):
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def generate():
        result = OutfitRecommender.recommend(image_base64, agent.get_user_info(), mime_type=mime_type)
        if not result.get('success'):
            yield send_event({'error': result.get('error', '穿搭推荐失败'), 'done': True})
            return

        recommendation = result.get('recommendation', '')
        image_url = result.get('image_url')

        chunk_size = 1
        for i in range(0, len(recommendation), chunk_size):
            yield send_event({'chunk': recommendation[i:i+chunk_size], 'done': False})

        if image_url:
            yield send_event({'image_url': image_url, 'done': False})

        if extra_payload:
            yield send_event({**extra_payload, 'done': False})

        yield send_event({'done': True})

        agent._conversation_history.append({
            'role': 'user',
            'content': '用户请求生成穿搭合成图'
        })
        agent._conversation_history.append({
            'role': 'assistant',
            'content': recommendation
        })
        agent._pending_outfit_recommendation = False

    return generate


@app.route('/api/products', methods=['GET'])
def products():
    """商品广场数据接口"""
    gender = request.args.get('gender', '')
    category = request.args.get('category', '')

    gender = filter_query_value(gender, {'男装': '男', '女装': '女'})
    category = filter_query_value(category, {'上衣': '上衣', '裤子': '裤子'})

    items = load_products_from_file()
    if gender:
        items = [item for item in items if item.get('gender') == gender]
    if category:
        items = [item for item in items if item.get('category') == category]

    return jsonify({'products': items})


@app.route('/query_order', methods=['POST'])
def query_order():
    """订单查询接口：根据手机号+姓名查询订单"""
    data = request.json or {}
    phone = str(data.get('phone', '')).strip()
    name = str(data.get('name', '')).strip()
    session_id = str(data.get('session_id', 'default')).strip() or 'default'

    if not phone or not name:
        return jsonify({'success': False, 'message': '请填写手机号和下单姓名'}), 400

    orders = load_orders_from_file()
    for order in orders:
        if str(order.get('phone', '')).strip() == phone and str(order.get('name', '')).strip() == name:
            agent = get_agent(session_id)
            _remember_order_context(agent, order)
            return jsonify({'success': True, 'order': order})

    return jsonify({'success': False, 'message': '未找到订单，请检查手机号和姓名'})


@app.route('/after_sales/submit', methods=['POST'])
def submit_after_sales():
    """提交售后工单接口。"""
    data = request.json or {}
    order_id = str(data.get('order_id', '')).strip()
    action = str(data.get('action', 'refund')).strip().lower()
    issue_type = str(data.get('issue_type', 'after_sales')).strip().lower()
    remark = str(data.get('remark', '')).strip()

    if action not in {'refund', 'exchange'}:
        return jsonify({'success': False, 'message': 'action 仅支持 refund/exchange'}), 400

    order = _find_order_by_order_id(order_id) if order_id else None
    if not isinstance(order, dict):
        return jsonify({'success': False, 'message': '未找到对应订单'}), 404

    ticket, created = _create_after_sales_ticket(order, issue_type, action, remark or '用户提交售后申请')
    return jsonify({
        'success': True,
        'created': created,
        'ticket': ticket,
        'status_text': _map_after_sales_status(ticket.get('status', 'submitted')),
        'refund_target': _refund_target_text(order),
    })


@app.route('/after_sales/query', methods=['GET'])
def query_after_sales():
    """查询售后工单接口。"""
    after_sales_id = str(request.args.get('after_sales_id', '')).strip()
    order_id = str(request.args.get('order_id', '')).strip()

    if after_sales_id:
        ticket = _find_after_sales_ticket_by_id(after_sales_id)
        if not isinstance(ticket, dict):
            return jsonify({'success': False, 'message': '未找到工单'}), 404
        return jsonify({'success': True, 'ticket': ticket, 'status_text': _map_after_sales_status(ticket.get('status', ''))})

    if order_id:
        ticket = _find_latest_after_sales_ticket(order_id)
        if not isinstance(ticket, dict):
            return jsonify({'success': False, 'message': '该订单暂无售后工单'}), 404
        return jsonify({'success': True, 'ticket': ticket, 'status_text': _map_after_sales_status(ticket.get('status', ''))})

    return jsonify({'success': False, 'message': '请提供 after_sales_id 或 order_id'}), 400


@app.route('/after_sales/update', methods=['POST'])
def update_after_sales():
    """更新售后工单状态接口（用于模拟流转）。"""
    data = request.json or {}
    after_sales_id = str(data.get('after_sales_id', '')).strip()
    status = str(data.get('status', '')).strip().lower()
    remark = str(data.get('remark', '')).strip()

    if not after_sales_id or not status:
        return jsonify({'success': False, 'message': '请提供 after_sales_id 和 status'}), 400

    ticket = _update_after_sales_status(after_sales_id, status, remark=remark)
    if not isinstance(ticket, dict):
        return jsonify({'success': False, 'message': '更新失败，工单不存在或状态非法'}), 400

    return jsonify({'success': True, 'ticket': ticket, 'status_text': _map_after_sales_status(ticket.get('status', ''))})

def get_agent(session_id: str):
    """获取或创建会话专属的智能体"""
    if session_id not in sessions:
        sessions[session_id] = FashionAIAgent(mode="auto", session_id=session_id)
    return sessions[session_id]


def warmup_default_session():
    """服务启动预热：创建默认会话并触发知识库与多智能体初始化。"""
    default_session_id = 'default'
    try:
        agent = get_agent(default_session_id)
        _ = agent.knowledge_base
        _ = agent.multi_agent
        logger.info("✅ 默认会话预热完成：知识库与多智能体已初始化")
    except Exception as e:
        logger.warning(f"⚠️ 默认会话预热失败: {e}")

@app.route('/chat', methods=['POST'])
def chat():
    """普通对话接口（非流式）"""
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    user_info = data.get('user_info', {})
    chat_settings = data.get('chat_settings', {})
    bypass_order_query_dialog = bool(data.get('bypass_order_query_dialog', False))
    
    agent = get_agent(session_id)
    agent.set_user_info(**user_info)
    agent.set_runtime_settings(chat_settings)
    intents = _resolve_dialog_intents(agent, message)

    if intents.get('coupon'):
        coupon_response = _handle_coupon_workflow(message)
        if coupon_response is not None:
            agent._conversation_history.append({'role': 'user', 'content': message})
            agent._conversation_history.append({'role': 'assistant', 'content': coupon_response})
            return jsonify({'response': coupon_response})

    if (
        not bypass_order_query_dialog
        and (intents.get('order_query') or getattr(agent, '_pending_order_query', False))
        and not _should_bypass_order_query_form(agent, message)
    ):
        order_dialog_response = _handle_order_query_dialog(agent, message)
        if order_dialog_response is not None:
            agent._conversation_history.append({'role': 'user', 'content': message})
            agent._conversation_history.append({'role': 'assistant', 'content': order_dialog_response})
            return jsonify({'response': order_dialog_response})

    order_ai_result = _try_handle_order_ai_assistant(agent, message)
    if order_ai_result is not None:
        order_ai_response, order_context = order_ai_result
        context_text = _build_order_context_inline(order_context)
        agent._conversation_history.append({'role': 'user', 'content': f'{message}\n订单上下文：{context_text}'})
        agent._conversation_history.append({'role': 'assistant', 'content': order_ai_response})
        return jsonify({'response': order_ai_response})

    _sync_order_context_from_message(agent, message)

    order_expedite_response = _handle_order_expedite_followup(agent, message)
    if order_expedite_response is not None:
        active_order = getattr(agent, '_active_order_context', None)
        context_text = _build_order_context_inline(active_order) if isinstance(active_order, dict) else ''
        user_content = f'{message}\n订单上下文：{context_text}' if context_text else message
        agent._conversation_history.append({'role': 'user', 'content': user_content})
        agent._conversation_history.append({'role': 'assistant', 'content': order_expedite_response})
        return jsonify({'response': order_expedite_response})

    if intents.get('order_followup') and not intents.get('after_sales_followup'):
        order_followup_response = _handle_order_context_followup(agent, message)
        if order_followup_response is not None:
            active_order = getattr(agent, '_active_order_context', None)
            context_text = _build_order_context_inline(active_order) if isinstance(active_order, dict) else ''
            user_content = f'{message}\n订单上下文：{context_text}' if context_text else message
            agent._conversation_history.append({'role': 'user', 'content': user_content})
            agent._conversation_history.append({'role': 'assistant', 'content': order_followup_response})
            return jsonify({'response': order_followup_response})

    after_sales_workflow_result = _handle_after_sales_workflow(agent, message, user_info) if intents.get('after_sales_followup') else None
    if after_sales_workflow_result is not None:
        after_sales_response, order_context = after_sales_workflow_result
        context_text = _build_order_context_inline(order_context)
        agent._conversation_history.append({'role': 'user', 'content': f'{message}\n订单上下文：{context_text}'})
        agent._conversation_history.append({'role': 'assistant', 'content': after_sales_response})
        return jsonify({'response': after_sales_response})

    if intents.get('outfit_followup'):
        cached = agent.get_last_image_context()
        if cached.get('image_base64'):
            result = OutfitRecommender.recommend(cached['image_base64'], agent.get_user_info(), mime_type=cached.get('mime_type', 'image/jpeg'))
            if result.get('success'):
                agent._pending_outfit_recommendation = False
                agent._conversation_history.append({'role': 'user', 'content': message})
                agent._conversation_history.append({'role': 'assistant', 'content': result.get('recommendation', '')})
                return jsonify(result)
        return jsonify({'response': '请先上传一张服装图片，我再帮您生成搭配合成图。'})

    if intents.get('size_followup'):
        size_context_message = _build_size_context_prompt(message)
        response = agent.chat(size_context_message, user_info)
        return jsonify({'response': response})
    
    response = agent.chat(message, user_info)
    return jsonify({'response': response})

@app.route('/chat/image', methods=['POST'])
def chat_image():
    """图片上传分析接口"""
    session_id = request.form.get('session_id', 'default')
    user_info_str = request.form.get('user_info', '{}')
    try:
        user_info = json.loads(user_info_str)
    except Exception:
        user_info = {}

    image_file = request.files.get('image')
    if image_file is None:
        return jsonify({'success': False, 'message': '未上传图片'}), 400

    filename = image_file.filename.lower()
    if not filename.endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'success': False, 'message': '仅支持 jpg/png 图片'}), 400

    image_bytes = image_file.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    mime_type = 'image/png' if filename.endswith('.png') else 'image/jpeg'

    agent = get_agent(session_id)
    agent.set_user_info(**user_info)

    structured_analysis = OutfitRecommender.analyze_clothing(image_base64, mime_type=mime_type)
    if structured_analysis:
        analysis = _build_image_advice_text(structured_analysis, agent.get_user_info())
    else:
        analysis = ImageAnalyzer.analyze(image_base64, mime_type=mime_type)
        if analysis:
            analysis = _append_outfit_followup_hint(analysis)

    if analysis is None:
        return jsonify({'success': False, 'message': '图片分析失败，请稍后重试'})

    agent.set_last_image_context(image_base64=image_base64, mime_type=mime_type, analysis=analysis)
    agent._pending_outfit_recommendation = True

    agent._conversation_history.append({
        'role': 'user',
        'content': '用户上传了一张服装图片，需要分析。'
    })
    agent._conversation_history.append({
        'role': 'assistant',
        'content': analysis
    })

    return jsonify({'success': True, 'analysis': analysis, 'response': analysis})

@app.route('/chat/image/stream', methods=['POST'])
def chat_image_stream():
    """图片上传分析流式接口"""
    session_id = request.form.get('session_id', 'default')
    user_info_str = request.form.get('user_info', '{}')
    try:
        user_info = json.loads(user_info_str)
    except Exception:
        user_info = {}

    image_file = request.files.get('image')
    if image_file is None:
        return jsonify({'success': False, 'message': '未上传图片'}), 400

    filename = image_file.filename.lower()
    if not filename.endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'success': False, 'message': '仅支持 jpg/png 图片'}), 400

    image_bytes = image_file.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    mime_type = 'image/png' if filename.endswith('.png') else 'image/jpeg'

    agent = get_agent(session_id)
    agent.set_user_info(**user_info)

    def send_event(payload: dict):
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def generate():
        structured_analysis = OutfitRecommender.analyze_clothing(image_base64, mime_type=mime_type)
        if structured_analysis:
            analysis = _build_image_advice_text(structured_analysis, agent.get_user_info())
        else:
            analysis = ImageAnalyzer.analyze(image_base64, mime_type=mime_type)
            if analysis:
                analysis = _append_outfit_followup_hint(analysis)

        if analysis is None:
            yield send_event({'error': '图片分析失败，请稍后重试', 'done': True})
            return

        agent.set_last_image_context(image_base64=image_base64, mime_type=mime_type, analysis=analysis)
        agent._pending_outfit_recommendation = True

        agent._conversation_history.append({
            'role': 'user',
            'content': '用户上传了一张服装图片，需要分析。'
        })
        agent._conversation_history.append({
            'role': 'assistant',
            'content': analysis
        })

        chunk_size = 1
        for i in range(0, len(analysis), chunk_size):
            yield send_event({'chunk': analysis[i:i+chunk_size], 'done': False})
        yield send_event({'done': True})

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """流式对话接口"""
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    user_info = data.get('user_info', {})
    chat_settings = data.get('chat_settings', {})
    bypass_order_query_dialog = bool(data.get('bypass_order_query_dialog', False))
    
    agent = get_agent(session_id)
    agent.set_user_info(**user_info)
    agent.set_runtime_settings(chat_settings)
    intents = _resolve_dialog_intents(agent, message)

    detected_emotion = '中性'
    try:
        detected_emotion = str(agent._emotion_analyzer.analyze(message).get('emotion', '中性') or '中性')
    except Exception:
        detected_emotion = '中性'

    if intents.get('coupon'):
        coupon_response = _handle_coupon_workflow(message)
        if coupon_response is not None:
            agent._conversation_history.append({'role': 'user', 'content': message})
            agent._conversation_history.append({'role': 'assistant', 'content': coupon_response})

            def generate_coupon_reply():
                return _stream_text_response(coupon_response, {'emotion': detected_emotion})()

            return Response(
                stream_with_context(generate_coupon_reply()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )

    if (
        not bypass_order_query_dialog
        and (intents.get('order_query') or getattr(agent, '_pending_order_query', False))
        and not _should_bypass_order_query_form(agent, message)
    ):
        order_dialog_response = _handle_order_query_dialog(agent, message)
        if order_dialog_response is not None:
            agent._conversation_history.append({'role': 'user', 'content': message})
            agent._conversation_history.append({'role': 'assistant', 'content': order_dialog_response})

            def generate_order_dialog():
                return _stream_text_response(order_dialog_response, {'emotion': detected_emotion})()

            return Response(
                stream_with_context(generate_order_dialog()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )

    order_ai_result = _try_handle_order_ai_assistant(agent, message)
    if order_ai_result is not None:
        order_ai_response, order_context = order_ai_result
        context_text = _build_order_context_inline(order_context)
        agent._conversation_history.append({'role': 'user', 'content': f'{message}\n订单上下文：{context_text}'})
        agent._conversation_history.append({'role': 'assistant', 'content': order_ai_response})

        def generate_order_ai_followup():
            return _stream_text_response(order_ai_response, {'emotion': detected_emotion})()

        return Response(
            stream_with_context(generate_order_ai_followup()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    _sync_order_context_from_message(agent, message)

    order_expedite_response = _handle_order_expedite_followup(agent, message)
    if order_expedite_response is not None:
        active_order = getattr(agent, '_active_order_context', None)
        context_text = _build_order_context_inline(active_order) if isinstance(active_order, dict) else ''
        user_content = f'{message}\n订单上下文：{context_text}' if context_text else message
        agent._conversation_history.append({'role': 'user', 'content': user_content})
        agent._conversation_history.append({'role': 'assistant', 'content': order_expedite_response})

        def generate_order_expedite_followup():
            return _stream_text_response(order_expedite_response, {'emotion': detected_emotion})()

        return Response(
            stream_with_context(generate_order_expedite_followup()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    if intents.get('order_followup') and not intents.get('after_sales_followup'):
        order_followup_response = _handle_order_context_followup(agent, message)
        if order_followup_response is not None:
            active_order = getattr(agent, '_active_order_context', None)
            context_text = _build_order_context_inline(active_order) if isinstance(active_order, dict) else ''
            user_content = f'{message}\n订单上下文：{context_text}' if context_text else message
            agent._conversation_history.append({'role': 'user', 'content': user_content})
            agent._conversation_history.append({'role': 'assistant', 'content': order_followup_response})

            def generate_order_context_followup():
                return _stream_text_response(order_followup_response, {'emotion': detected_emotion})()

            return Response(
                stream_with_context(generate_order_context_followup()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )

    after_sales_workflow_result = _handle_after_sales_workflow(agent, message, user_info) if intents.get('after_sales_followup') else None
    if after_sales_workflow_result is not None:
        after_sales_response, order_context = after_sales_workflow_result
        context_text = _build_order_context_inline(order_context)
        agent._conversation_history.append({'role': 'user', 'content': f'{message}\n订单上下文：{context_text}'})
        agent._conversation_history.append({'role': 'assistant', 'content': after_sales_response})

        def generate_after_sales_followup():
            return _stream_text_response(after_sales_response, {'emotion': detected_emotion})()

        return Response(
            stream_with_context(generate_after_sales_followup()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    if intents.get('outfit_followup'):
        cached = agent.get_last_image_context()
        if cached.get('image_base64'):
            def generate_outfit():
                return _stream_outfit_recommendation(
                    agent,
                    cached['image_base64'],
                    cached.get('mime_type', 'image/jpeg'),
                    user_info,
                    {'emotion': detected_emotion}
                )()

            return Response(
                stream_with_context(generate_outfit()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )

        def generate_hint():
            return _stream_text_response('请先上传一张服装图片，我再帮您生成搭配合成图。', {'emotion': detected_emotion})()

        return Response(
            stream_with_context(generate_hint()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    if intents.get('size_followup'):
        size_context_message = _build_size_context_prompt(message)

        def generate_size_followup():
            return _stream_text_response(agent.chat(size_context_message, user_info), {'emotion': detected_emotion})()

        return Response(
            stream_with_context(generate_size_followup()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
    
    def generate():
        for chunk in agent.chat_stream(message, user_info):
            yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
        yield f"data: {json.dumps({'emotion': detected_emotion, 'done': False})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/chat/outfit_recommend', methods=['POST'])
def outfit_recommend():
    """穿搭推荐接口"""
    data = request.json
    image_base64 = data.get('image_base64', '')
    session_id = data.get('session_id', 'default')
    user_info = data.get('user_info', {})
    mime_type = data.get('mime_type', 'image/jpeg')

    if not image_base64:
        return jsonify({'success': False, 'error': '缺少图片数据'}), 400

    # 获取会话代理（可选，用于记录历史）
    agent = get_agent(session_id)
    agent.set_user_info(**user_info)

    # 调用穿搭推荐器，使用会话中记住的用户信息
    result = OutfitRecommender.recommend(image_base64, agent.get_user_info(), mime_type=mime_type)

    # 将推荐结果添加到会话历史
    if result['success']:
        agent._conversation_history.append({
            'role': 'user',
            'content': '用户请求穿搭推荐'
        })
        agent._conversation_history.append({
            'role': 'assistant',
            'content': result['recommendation']
        })

    return jsonify(result)

@app.route('/chat/outfit_recommend/stream', methods=['POST'])
def outfit_recommend_stream():
    """穿搭推荐流式输出接口"""
    data = request.json
    image_base64 = data.get('image_base64', '')
    session_id = data.get('session_id', 'default')
    user_info = data.get('user_info', {})
    mime_type = data.get('mime_type', 'image/jpeg')

    if not image_base64:
        return jsonify({'success': False, 'error': '缺少图片数据'}), 400

    agent = get_agent(session_id)
    agent.set_user_info(**user_info)

    def send_event(payload: dict):
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def generate():
        result = OutfitRecommender.recommend(image_base64, agent.get_user_info(), mime_type=mime_type)
        if not result.get('success'):
            yield send_event({'error': result.get('error', '穿搭推荐失败'), 'done': True})
            return

        recommendation = result.get('recommendation', '')
        image_url = result.get('image_url')

        chunk_size = 1
        for i in range(0, len(recommendation), chunk_size):
            yield send_event({'chunk': recommendation[i:i+chunk_size], 'done': False})

        if image_url:
            yield send_event({'image_url': image_url, 'done': False})

        yield send_event({'done': True})

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/chat/clear', methods=['POST'])
def clear_chat():
    """清除会话历史"""
    data = request.get_json(silent=True) or {}
    session_id = data.get('session_id', 'default')
    
    if session_id in sessions:
        sessions[session_id].clear_history()
        return jsonify({'success': True, 'message': '对话已清空'})
    return jsonify({'success': False, 'message': '会话不存在'})

if __name__ == '__main__':
    # 关闭调试模式的自动重载，避免 BGE/torch 模块触发 watcher 重启
    warmup_default_session()
    app.run(
        host=os.getenv('FASHIONAI_HOST', '127.0.0.1'),
        port=int(os.getenv('FASHIONAI_PORT', '5000')),
        debug=False,
        use_reloader=False,
    )
