import contextlib
import io

import pandas as pd
import streamlit as st

from conversation import Conversation
from exceptions import InvalidPromptError
from manager import ModelManager
from models import GPTModel, LanguageModel, LlamaModel
from pricing import PricingPlan

st.set_page_config(page_title="LLM Manager", page_icon="🤖", layout="wide")

# Ready-made pricing plans: plan name -> price per 1000 tokens
PLANS = {
    "Free": 0.0,
    "Standard": 0.02,
    "Premium": 0.05,
}

if "manager" not in st.session_state:
    st.session_state.manager = ModelManager()
    st.session_state.conversations = {}

manager = st.session_state.manager
conversations = st.session_state.conversations

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("Add model")
    model_type = st.radio("Type", ["GPT", "Llama"], horizontal=True)

    name = st.text_input("Model name", value="GPT-Model" if model_type == "GPT" else "Llama-Model")
    provider = st.text_input("Provider", value="OpenAI" if model_type == "GPT" else "Meta")
    max_tokens = st.number_input("Max tokens", min_value=1, value=4096, step=100)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

    if model_type == "GPT":
        api_key = st.text_input("API key", type="password")
    else:
        model_path = st.text_input("Model path", value="/models/llama")
        quantization = st.text_input("Quantization", value="4bit")

    plan_name = st.selectbox("Pricing plan", list(PLANS.keys()), index=1)
    price = PLANS[plan_name]
    st.caption(f"Price per 1000 tokens: ${price}")

    if st.button("Add"):
        if not name.strip():
            st.error("Model name is required")
        elif name in manager.models:
            st.error("A model with this name already exists")
        else:
            plan = PricingPlan(plan_name, price)
            if model_type == "GPT":
                model = GPTModel(name, provider, int(max_tokens), temperature, api_key, plan)
            else:
                model = LlamaModel(name, provider, int(max_tokens), temperature,
                                   model_path, quantization, plan)
            manager.add_model(model)
            st.success(f"Added {name}")

    st.divider()
    st.caption(f"Total models created: {LanguageModel.total_models}")

# ---------------------------------------------------------------- Main
st.title("LLM Model Manager")

if not manager.models:
    st.info("Add a model from the sidebar to get started.")
    st.stop()

names = list(manager.models.keys())
tab_chat, tab_models, tab_cost, tab_select, tab_oop = st.tabs(
    ["Chat", "Models", "Cost", "Select", "OOP Concepts"]
)


def run(fn):
    """Run fn, capture what it prints, its return value, and any exception."""
    buffer = io.StringIO()
    result, error = None, None
    with contextlib.redirect_stdout(buffer):
        try:
            result = fn()
        except Exception as e:  # shown to the viewer on purpose
            error = e
    return buffer.getvalue().strip(), result, error


def show(fn):
    """Run fn and display printed output, returned value, or the exception."""
    printed, result, error = run(fn)
    if error is not None:
        st.error(f"{type(error).__name__}: {error}")
    else:
        if printed:
            st.code(printed)
        if result is not None:
            st.code(repr(result))

# ---- Chat
with tab_chat:
    chosen = st.selectbox("Model", names, key="chat_model")
    model = manager.models[chosen]
    if chosen not in conversations:
        conversations[chosen] = Conversation(f"conv-{chosen}", model)
    conversation = conversations[chosen]

    for message in conversation.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Type a message")
    if prompt is not None:
        try:
            conversation.send_message(prompt)
            st.rerun()
        except InvalidPromptError as e:
            st.error(f"Error sending message: {e}")

    if st.button("Clear conversation"):
        conversation.messages.clear()
        st.rerun()

# ---- Models
with tab_models:
    rows = [
        {
            "name": m.model_name,
            "type": type(m).__name__,
            "provider": m.provider,
            "max_tokens": len(m),
            "temperature": m.temperature,
            "active": m.is_active,
        }
        for m in manager.models.values()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    selected = st.selectbox("Select model", names, key="edit_model")
    target = manager.models[selected]

    col1, col2 = st.columns(2)
    with col1:
        new_temp = st.number_input("New temperature", value=float(target.temperature), step=0.1)
        if st.button("Apply"):
            if LanguageModel.validate_temperature(new_temp):
                target.temperature = new_temp
                st.rerun()
            else:
                st.error("Temperature must be between 0 and 2.")

        active = st.checkbox("Active", value=target.is_active, key=f"active_{selected}")
        if active != target.is_active:
            target.is_active = active
            st.rerun()
    with col2:
        st.write("")
        st.write("")
        if st.button("Delete model", type="primary"):
            manager.remove_model(selected)
            conversations.pop(selected, None)
            st.rerun()

    with st.expander("repr / str"):
        st.code(repr(target))
        st.code(str(target))

# ---- Cost
with tab_cost:
    chosen = st.selectbox("Model", names, key="cost_model")
    text = st.text_area("Prompt")
    if st.button("Calculate cost"):
        if not text.strip():
            st.error("Prompt must be a non-empty string.")
        else:
            target = manager.models[chosen]
            tokens = target.tokenizer.count_tokens(text)
            cost = target.pricing_plan.calculate_cost(tokens)
            c1, c2 = st.columns(2)
            c1.metric("Input tokens", tokens)
            c2.metric("Cost", f"${cost:.5f}")

# ---- Select
with tab_select:
    providers = sorted({m.provider for m in manager.models.values()})
    provider_choice = st.selectbox("Provider", providers)
    required = st.number_input("Required tokens", min_value=1, value=500, step=100)
    if st.button("Find model"):
        found = manager.select_model(provider_choice, int(required))
        if found:
            st.success(f"Matching model: {found.model_name}")
            st.code(repr(found))
        else:
            st.warning("No model matches these requirements")

# ---- OOP Concepts
with tab_oop:
    st.caption("Click a button to see each OOP concept working live.")
    oop_name = st.selectbox("Model to use", names, key="oop_model")
    m = manager.models[oop_name]
    others = [x for x in manager.models.values() if x is not m]

    st.subheader("1. Abstraction (ABC)")
    st.write("`LanguageModel` is abstract, so we cannot create an object from it.")
    if st.button("Try LanguageModel(...)"):
        show(lambda: LanguageModel("x", "y", 100, 0.5, PricingPlan("Free", 0.0)))

    st.subheader("2. Encapsulation (private attribute + property)")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Read model.__temperature directly"):
            show(lambda: getattr(m, "__temperature"))
    with c2:
        if st.button("Read model.temperature (property)"):
            show(lambda: m.temperature)
    bad_temp = st.number_input("Try setting temperature to", value=5.0, step=0.5, key="bad_temp")
    if st.button("Set temperature (setter validates)"):
        def try_set():
            m.temperature = bad_temp
            return m.temperature
        show(try_set)

    st.subheader("3. Polymorphism")
    st.write("Same method `generate_response`, different behavior per class.")
    poly_prompt = st.text_input("Prompt", value="Explain OOP", key="poly_prompt")
    if st.button("Send to every model"):
        for model_obj in manager.models.values():
            st.write(f"**{model_obj.model_name}** ({type(model_obj).__name__})")
            show(lambda mo=model_obj: mo.generate_response(poly_prompt))

    st.subheader("4. Custom exception")
    if st.button("Send an empty prompt"):
        show(lambda: m.generate_response("   "))

    st.subheader("5. Magic methods")
    if st.button("Show str / repr / len"):
        st.code(f"str(model)  -> {m}\nrepr(model) -> {m!r}\nlen(model)  -> {len(m)}")
    if st.button("Compare with another model (__eq__)"):
        if others:
            show(lambda: m == others[0])
        else:
            st.info("Add a second model to compare.")
    if st.button("Compare with a number (__eq__ -> NotImplemented)"):
        show(lambda: m == 5)

    st.subheader("6. Class method and static method")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("LanguageModel.get_total_models()"):
            show(lambda: LanguageModel.get_total_models())
    with c2:
        static_value = st.number_input("validate_temperature(", value=3.0, step=0.5, key="static_val")
        if st.button("Run validate_temperature"):
            show(lambda: LanguageModel.validate_temperature(static_value))

    st.subheader("7. Multiple inheritance (Mixin) and MRO")
    if st.button("Show MRO of this model"):
        st.code(" -> ".join(c.__name__ for c in type(m).__mro__))
    if st.button("Call generate_response and show the log"):
        show(lambda: m.generate_response("hello"))

    st.subheader("8. Composition vs aggregation")
    if st.button("Show the parts of this model"):
        st.code(
            f"tokenizer    -> {type(m.tokenizer).__name__}  (composition: created inside the model)\n"
            f"pricing_plan -> {type(m.pricing_plan).__name__}  (aggregation: passed in from outside)"
        )

    st.subheader("9. display_info")
    if st.button("Run display_info()"):
        show(lambda: m.display_info())