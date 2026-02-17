import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.set_page_config(
    page_title="Code Snippet Generator",
    page_icon="🤖",
    layout="wide"
)

# ------------------------------
# Load Model (Cached)
# ------------------------------
@st.cache_resource
def load_model():
    model = T5ForConditionalGeneration.from_pretrained("t5_finetuned_model")
    tokenizer = T5Tokenizer.from_pretrained("t5_finetuned_model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# ------------------------------
# Generate Function
# ------------------------------
def generate_code(query):
    input_text = "generate code: " + query.lower()
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------------------
# UI
# ------------------------------
st.title("🤖 Code Snippet Generator")
st.markdown("Generate code snippets using your fine-tuned T5 model.")

st.markdown("### Example Queries")
st.markdown("""
1. How to handle events in JavaScript?  
2. How to use a lambda function in Python?  
3. How to create a shell script?
""")

query = st.text_area(
    "Enter your programming query:",
    height=120,
    placeholder="Example: How to handle events in JavaScript?"
)

col1, col2 = st.columns([1, 1])

generate_btn = col1.button("🚀 Generate")
like_btn = col2.button("👍 Like")

# ------------------------------
# Generate Output
# ------------------------------
if generate_btn:
    if query.strip() == "":
        st.warning("⚠ Please enter a query.")
    else:
        with st.spinner("Generating code..."):
            try:
                result = generate_code(query)

                st.success("✅ Code Generated Successfully")

                if "Code:" in result:
                    parts = result.split("Code:")
                    meta = parts[0].strip()
                    code = parts[1].strip()

                    if meta:
                        st.markdown("### Details")
                        st.info(meta)

                    st.markdown("### Generated Code")
                    st.code(code)

                else:
                    st.markdown("### Generated Code")
                    st.code(result)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ------------------------------
# Like Button Feedback
# ------------------------------
if like_btn:
    st.toast("🔥 Thanks for liking the model!")