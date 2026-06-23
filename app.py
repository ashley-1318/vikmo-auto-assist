"""
app.py - Streamlit UI for VIKMO AI Dealer Assistant

Features:
- ChatGPT-style chat interface with message bubbles
- Sidebar: low stock alerts, order history, session info
- Product cards with stock indicators
- Tool execution logs (collapsible)
- Order confirmation panels
- RAG retrieval results panel
"""

import os
import json
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="VIKMO AI Dealer Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Styling ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0f1117; color: #e0e0e0; }

.chat-user {
    background: linear-gradient(135deg, #1e3a5f, #2563eb);
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 20%;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3);
}
.chat-assistant {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    color: #e2e8f0;
    border: 1px solid #2d3748;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 20% 8px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.product-card {
    background: linear-gradient(135deg, #1a1f2e, #1e2538);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    transition: transform 0.2s;
}
.product-card:hover { transform: translateY(-2px); border-color: #3b82f6; }
.sku-badge {
    background: #1e3a5f;
    color: #60a5fa;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.8em;
    font-weight: 600;
    font-family: monospace;
}
.stock-ok { color: #4ade80; font-weight: 600; }
.stock-low { color: #facc15; font-weight: 600; }
.stock-out { color: #f87171; font-weight: 600; }
.alert-critical {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.85em;
}
.alert-warning {
    background: linear-gradient(135deg, #78350f, #92400e);
    border: 1px solid #f59e0b;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.85em;
}
.order-card {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
}
.tool-log {
    background: #0d1117;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 12px;
    font-family: monospace;
    font-size: 0.8em;
    color: #a0aec0;
}
.header-gradient {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a1a3e 50%, #0f2027 100%);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    text-align: center;
    border: 1px solid #2d3748;
}
.metric-box {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}
.vision-card {
    background: linear-gradient(135deg, #1a1f2e, #1e2538);
    border: 1px solid #7c3aed;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
}
.confidence-high  { color: #4ade80; font-weight: 700; }
.confidence-med   { color: #facc15; font-weight: 700; }
.confidence-low   { color: #f87171; font-weight: 700; }
.vision-badge {
    background: #4c1d95;
    color: #c4b5fd;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78em;
    font-weight: 600;
    display: inline-block;
    margin: 2px 3px;
}
stButton button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ───────────────────────
def init_session():
    if "memory" not in st.session_state:
        from assistant.memory import ConversationMemory
        st.session_state.memory = ConversationMemory()
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages_display" not in st.session_state:
        st.session_state.messages_display = []
    if "agent_ready" not in st.session_state:
        st.session_state.agent_ready = False
    if "tool_logs" not in st.session_state:
        st.session_state.tool_logs = []
    if "orders" not in st.session_state:
        st.session_state.orders = []
    if "vision_result" not in st.session_state:
        st.session_state.vision_result = None   # Last vision identification result
    if "last_uploaded_name" not in st.session_state:
        st.session_state.last_uploaded_name = None  # Track file name to avoid re-running


@st.cache_resource(show_spinner="🔧 Loading AI engine (first run may take ~60s)...")
def load_agent():
    from assistant.agent import DealerAgent
    agent = DealerAgent()
    agent.initialize()
    return agent


def render_product_card(product: dict):
    stock = product.get("stock", 0)
    if stock == 0:
        stock_html = '<span class="stock-out">❌ Out of Stock</span>'
    elif product.get("low_stock"):
        stock_html = f'<span class="stock-low">⚠️ Low Stock: {stock} units</span>'
    else:
        stock_html = f'<span class="stock-ok">✅ In Stock: {stock} units</span>'

    st.markdown(f"""
    <div class="product-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <strong style="color:#e2e8f0;">{product.get("name","")}</strong>
            <span class="sku-badge">{product.get("sku","")}</span>
        </div>
        <div style="color:#94a3b8; font-size:0.85em; margin-bottom:6px;">
            🏷️ {product.get("brand","")} | 📂 {product.get("category","")}
        </div>
        <div style="color:#64748b; font-size:0.8em; margin-bottom:8px;">
            🚗 {product.get("vehicle_fitment","")[:80]}{"..." if len(str(product.get("vehicle_fitment","")))>80 else ""}
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong style="color:#3b82f6; font-size:1.1em;">₹{product.get("price_inr",0):,.0f}</strong>
            {stock_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_order_card(order: dict):
    st.markdown(f"""
    <div class="order-card">
        <div style="font-size:1.1em; font-weight:700; color:#10b981; margin-bottom:8px;">
            ✅ Order Confirmed: {order.get("order_id","")}
        </div>
        <div style="color:#6ee7b7; margin-bottom:4px;">
            👤 Dealer: {order.get("dealer_name","")}
        </div>
        <div style="color:#a7f3d0; font-size:0.85em; margin-bottom:8px;">
            🕐 {order.get("timestamp","")[:19]} UTC
        </div>
    </div>
    """, unsafe_allow_html=True)

    if order.get("items"):
        cols = st.columns([3, 1, 1, 1])
        cols[0].markdown("**Item**")
        cols[1].markdown("**Qty**")
        cols[2].markdown("**Unit Price**")
        cols[3].markdown("**Line Total**")
        for item in order["items"]:
            c0, c1, c2, c3 = st.columns([3, 1, 1, 1])
            c0.write(f"{item['name']} ({item['sku']})")
            c1.write(str(item["quantity"]))
            c2.write(f"₹{item['unit_price_inr']:,.0f}")
            c3.write(f"₹{item['line_total_inr']:,.0f}")

    st.markdown(f"""
    <div style="margin-top:10px; padding:10px; background:#065f46; border-radius:8px; text-align:right;">
        <strong style="color:#10b981; font-size:1.1em;">Total: ₹{order.get("total_amount_inr",0):,.2f}</strong>
    </div>
    """, unsafe_allow_html=True)

    if order.get("warnings"):
        for w in order["warnings"]:
            st.warning(w)


def render_sidebar(agent):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:16px 0;">
            <div style="font-size:2em;">🔧</div>
            <div style="font-weight:700; font-size:1.1em; color:#3b82f6;">VIKMO Assistant</div>
            <div style="color:#64748b; font-size:0.8em;">AI Dealer Portal</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Session info
        memory = st.session_state.memory
        vehicle_ctx = memory.get_vehicle_context()
        primary_sku = memory.get_primary_sku()

        st.markdown("**📊 Session Context**")
        col1, col2 = st.columns(2)
        col1.metric("Messages", len(memory.messages))
        col2.metric("Orders", len(memory.order_history))

        if vehicle_ctx:
            st.info(f"🚗 **Vehicle:** {vehicle_ctx}")
        if primary_sku:
            st.info(f"🔩 **Last SKU:** {primary_sku}")

        st.divider()

        # Low stock alerts
        st.markdown("**🚨 Low Stock Alerts**")
        if agent:
            try:
                alerts = agent.get_low_stock_alerts(threshold=15)
                if alerts:
                    for alert in alerts[:8]:
                        css_class = "alert-critical" if alert["alert_level"] == "CRITICAL" else "alert-warning"
                        icon = "🔴" if alert["alert_level"] == "CRITICAL" else "🟡"
                        st.markdown(f"""
                        <div class="{css_class}">
                            {icon} <strong>{alert["sku"]}</strong><br>
                            {alert["name"][:35]}...<br>
                            Stock: <strong>{alert["stock"]}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ All stock levels healthy")
            except Exception:
                st.info("Loading alerts...")

        st.divider()

        # Order history
        if st.session_state.orders:
            st.markdown("**📋 Order History**")
            for order in reversed(st.session_state.orders[-5:]):
                st.markdown(f"""
                <div style="background:#1a2538; border:1px solid #2d3748; border-radius:8px; padding:8px; margin:4px 0; font-size:0.82em;">
                    <strong style="color:#10b981;">{order["order_id"]}</strong><br>
                    {order["dealer_name"]}<br>
                    <span style="color:#94a3b8;">₹{order["total_amount_inr"]:,.0f}</span>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        if st.button("🗑️ New Conversation", use_container_width=True):
            st.session_state.memory.clear()
            st.session_state.messages_display = []
            st.session_state.tool_logs = []
            st.session_state.orders = []
            st.rerun()

        # Quick commands
        st.markdown("**💡 Quick Prompts**")
        quick_prompts = [
            "Show tyres for KTM Duke 390",
            "Check stock BRK-1003",
            "Find parts for Royal Enfield Himalayan",
            "I need brake pads for Bajaj Pulsar NS200",
        ]
        for prompt in quick_prompts:
            if st.button(f"▶ {prompt[:35]}", key=f"qp_{prompt[:20]}", use_container_width=True):
                st.session_state["quick_input"] = prompt
                st.rerun()


def main():
    init_session()

    # Header
    st.markdown("""
    <div class="header-gradient">
        <h1 style="margin:0; color:#e2e8f0; font-size:1.8em;">🔧 VIKMO AI Dealer Assistant</h1>
        <p style="margin:6px 0 0; color:#94a3b8;">Powered by Llama 3.3 70B · RAG · Tool Calling</p>
    </div>
    """, unsafe_allow_html=True)

    # Load agent
    if not st.session_state.agent_ready:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            st.error("❌ GROQ_API_KEY not found. Please set it in your .env file.")
            st.code('GROQ_API_KEY=your_key_here', language="bash")
            st.stop()

        with st.spinner("🚀 Initializing AI engine..."):
            st.session_state.agent = load_agent()
            st.session_state.agent_ready = True

    agent = st.session_state.agent
    render_sidebar(agent)

    # ─── Main Content Layout ─────────────────────────────
    chat_col, info_col = st.columns([2, 1])

    with chat_col:
        st.markdown("### 💬 Chat")

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages_display:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-assistant">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

                    # Show order card if order was placed
                    if msg.get("order"):
                        render_order_card(msg["order"])

                    # Show tool logs (collapsible)
                    if msg.get("tool_calls"):
                        with st.expander(f"🛠️ Tool Calls ({len(msg['tool_calls'])})", expanded=False):
                            for tc in msg["tool_calls"]:
                                st.markdown(f'<div class="tool-log">⚙️ <b>{tc["tool"]}</b>({json.dumps(tc["args"])})<br>→ {json.dumps(tc["result"])[:300]}...</div>', unsafe_allow_html=True)

        st.divider()

        # Chat input
        quick_input = st.session_state.pop("quick_input", None)
        default_val = quick_input if quick_input else ""

        with st.form("chat_form", clear_on_submit=True):
            col_input, col_btn = st.columns([5, 1])
            with col_input:
                user_input = st.text_input(
                    "Ask about parts, stock, vehicle fitment or place an order...",
                    value=default_val,
                    label_visibility="collapsed",
                    placeholder="e.g. Show brake pads for KTM Duke 390",
                )
            with col_btn:
                submitted = st.form_submit_button("Send 🚀", use_container_width=True)

        if submitted and user_input.strip():
            with st.spinner("🤔 Thinking..."):
                try:
                    result = agent.chat(user_input.strip(), st.session_state.memory)

                    # Store for display
                    st.session_state.messages_display.append({
                        "role": "user",
                        "content": user_input.strip(),
                    })
                    msg_data = {
                        "role": "assistant",
                        "content": result["response"],
                        "tool_calls": result.get("tool_calls", []),
                        "retrieved_products": result.get("retrieved_products", []),
                        "order": result.get("order"),
                    }
                    st.session_state.messages_display.append(msg_data)
                    st.session_state.tool_logs.extend(result.get("tool_calls", []))

                    if result.get("order"):
                        st.session_state.orders.append(result["order"])

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.exception("Chat error")

            st.rerun()

    with info_col:
        st.markdown("### 🔍 Retrieved Products")

        # Show retrieved products from last assistant message
        last_assistant = None
        for msg in reversed(st.session_state.messages_display):
            if msg["role"] == "assistant" and msg.get("retrieved_products"):
                last_assistant = msg
                break

        if last_assistant and last_assistant.get("retrieved_products"):
            products = last_assistant["retrieved_products"]
            st.caption(f"Top {len(products)} semantically relevant products:")
            for prod in products[:5]:
                render_product_card(prod)
        else:
            st.markdown("""
            <div style="background:#1a1f2e; border:1px dashed #2d3748; border-radius:12px; padding:32px; text-align:center; color:#64748b;">
                <div style="font-size:2em; margin-bottom:8px;">🔎</div>
                <div>Product matches will appear here when you ask a question.</div>
            </div>
            """, unsafe_allow_html=True)

        # ─── Image Part Identification ─────────────────────────
        st.markdown("### 📷 Part Identification")
        st.caption("Upload a photo of any automotive part to identify it and find matches in the catalogue.")

        uploaded = st.file_uploader(
            "Upload part image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="part_image_uploader",
        )

        if uploaded is not None:
            # Show the image
            st.image(uploaded, caption=uploaded.name, use_container_width=True)

            # Only run identification when a NEW image is uploaded
            if st.session_state.last_uploaded_name != uploaded.name:
                st.session_state.last_uploaded_name = uploaded.name
                st.session_state.vision_result = None  # Clear old result

                with st.spinner("🔍 Identifying part using Groq Vision AI..."):
                    try:
                        image_bytes = uploaded.getvalue()
                        # Determine MIME type from file extension
                        ext = uploaded.name.rsplit(".", 1)[-1].lower()
                        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                                    "png": "image/png", "webp": "image/webp"}
                        mime_type = mime_map.get(ext, "image/jpeg")

                        result = agent.identify_part_from_image(image_bytes, mime_type)
                        st.session_state.vision_result = result
                    except Exception as e:
                        st.session_state.vision_result = {"error": str(e)}

            # ── Render vision result ──────────────────────────
            vr = st.session_state.vision_result
            if vr:
                if vr.get("error"):
                    st.error(f"❌ {vr['error']}")
                else:
                    v = vr.get("vision_result", {})
                    confidence = v.get("confidence", "Low")
                    conf_class = {
                        "High": "confidence-high",
                        "Medium": "confidence-med",
                        "Low": "confidence-low",
                    }.get(confidence, "confidence-low")
                    conf_icon = {"High": "✅", "Medium": "⚠️", "Low": "❓"}.get(confidence, "❓")

                    brand_html = (
                        f'<span class="vision-badge">🏷️ {v["brand_if_visible"]}</span>'
                        if v.get("brand_if_visible")
                        else ""
                    )
                    condition_html = (
                        f'<span class="vision-badge">🔧 {v.get("condition","Unknown")}</span>'
                    )
                    cat_html = (
                        f'<span class="vision-badge">📂 {v.get("category","Unknown")}</span>'
                    )

                    st.markdown(f"""
                    <div class="vision-card">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px;">
                            <strong style="color:#c4b5fd; font-size:1.05em;">🔍 {v.get("part_name","Unknown Part")}</strong>
                            <span class="{conf_class}">{conf_icon} {confidence} Confidence</span>
                        </div>
                        <div style="margin-bottom:8px;">
                            {cat_html} {brand_html} {condition_html}
                        </div>
                        <div style="color:#94a3b8; font-size:0.88em; line-height:1.5;">
                            {v.get("description","")}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Catalogue matches
                    matches = vr.get("matched_products", [])
                    if matches:
                        st.markdown("**🛒 Catalogue Matches:**")
                        for prod in matches[:3]:
                            render_product_card(prod)

                        # One-click: send identification to chat
                        part_name = v.get("part_name", "this part")
                        search_q = v.get("search_query", part_name)
                        if st.button(
                            f"💬 Ask about {part_name} in chat",
                            key="vision_to_chat",
                            use_container_width=True,
                        ):
                            st.session_state["quick_input"] = (
                                f"Show me {search_q} options and check stock"
                            )
                            st.rerun()
                    else:
                        st.info("No exact catalogue matches found. Try describing the part in chat for a deeper search.")


if __name__ == "__main__":
    main()
