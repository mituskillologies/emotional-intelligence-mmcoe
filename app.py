import streamlit as st
from transformers import pipeline
import ollama

st.set_page_config(page_title="Emotionally Adaptive AI (Local Mode)", layout="wide")

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_sentiment_model()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sentiment_history" not in st.session_state:
    st.session_state.sentiment_history = []
if "de_escalation_rate" not in st.session_state:
    st.session_state.de_escalation_rate = 0.0

def analyze_student_state(user_input):
    result = sentiment_analyzer(user_input)[0]
    label = result['label']
    score = result['score']
    
    is_frustrated = True if label == 'NEGATIVE' and score > 0.85 else False
    normalized_score = -score if label == 'NEGATIVE' else score
    
    return is_frustrated, normalized_score

def generate_system_prompt(is_frustrated):
    base_knowledge = "You are a highly capable Python programming teaching assistant."
    
    if is_frustrated:
        persona = """ 
        [SYSTEM DIRECTIVE: HIGH COGNITIVE LOAD DETECTED]
        The student is experiencing severe frustration. 
        Your rules: 
        1. Validate their frustration first.
        2. Significantly slow the conversational pacing. 
        3. Break the technical problem down into the smallest logical steps. 
        4. Use a warm, highly encouraging, and supportive tone.
        """
    else:
        persona = """
        [SYSTEM DIRECTIVE: NORMAL ENGAGEMENT]
        The student is engaged. Provide clear, concise, and direct technical answers.
        """
    return f"{base_knowledge} {persona}"

def calculate_de_escalation(history):
    if len(history) < 2:
        return 0.0
    initial_state = history[0]
    current_state = history[-1]
    return current_state - initial_state

st.title("Capstone Demo: Adaptive AI Mentor (Local Model)")
st.markdown("Running 100% locally using tinyllama model.")

with st.sidebar:
    st.header("Live Engineering KPIs")
    st.markdown("---")
    
    current_state = "Neutral/Positive"
    if st.session_state.sentiment_history:
        latest_score = st.session_state.sentiment_history[-1]
        if latest_score < -0.85:
            current_state = "High Frustration Detected"
            
    st.metric(label="Current User State", value=current_state)
    
    st.metric(
        label="De-escalation Rate (Delta)", 
        value=f"{st.session_state.de_escalation_rate:+.2f}",
        delta="Positive indicates calming influence",
        delta_color="normal"
    )
    
    st.markdown("### Sentiment History Pipeline")
    st.line_chart(st.session_state.sentiment_history)
    
    if st.button("Reset Session Memory"):
        st.session_state.messages = []
        st.session_state.sentiment_history = []
        st.session_state.de_escalation_rate = 0.0
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your programming question or frustration..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    is_frustrated, normalized_score = analyze_student_state(prompt)
    st.session_state.sentiment_history.append(normalized_score)
    st.session_state.de_escalation_rate = calculate_de_escalation(st.session_state.sentiment_history)
    
    dynamic_system_prompt = generate_system_prompt(is_frustrated)
    temp_setting = 0.4 if is_frustrated else 0.7 
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if is_frustrated:
            st.caption("*(System Shift: Adjusting Persona to Empathetic & Lowering Temperature)*")
            
        try:
            api_messages = [{"role": "system", "content": dynamic_system_prompt}]
            for m in st.session_state.messages:
                api_messages.append({"role": m["role"], "content": m["content"]})
                
            # Enable the stream=True parameter
            stream = ollama.chat(
                model='tinyllama',
                messages=api_messages,
                options={'temperature': temp_setting},
                stream=True # This is the crucial addition
            )
            
            full_response = ""
            # Iterate through the stream and update the UI in real-time
            for chunk in stream:
                full_response += chunk['message']['content']
                # Add a blinking cursor effect for a polished look
                message_placeholder.markdown(full_response + "▌") 
            
            # Final update to remove the cursor once generation is complete
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            full_response = f"Local Inference Error. Details: {str(e)}"
            message_placeholder.error(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()
